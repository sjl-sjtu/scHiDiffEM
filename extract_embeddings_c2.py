"""
Extract per-cell embeddings from C2 EM checkpoints.

Per cell we build the rotation-invariant feature log1p(softplus(ZZᵀ+B)) on the
band, then reduce to a cell embedding.  Two fusion strategies across chromosomes:

  --fusion two_stage   per-chrom PCA(--per_chrom_dim) → concat → global
                       PCA(--embed_dim).  This is the scHiCluster DipC recipe.
                       Cheap, but compresses each chrom to per_chrom_dim BEFORE
                       fusion (an early bottleneck).

  --fusion joint       ONE PCA over the concatenation of ALL chromosomes'
                       features, computed memory-frugally via the N×N Gram
                       matrix (Σ_chrom X_c X_cᵀ).  No per-chrom bottleneck —
                       closer in spirit to Fast-Higashi's shared cell factor.
                       Each chrom is variance-normalized so none dominates.

Defaults are the sweep winner (beats Higashi on this data): --fusion two_stage,
--embed_dim 30, --center 0, --whiten 0, --l2norm 1.  The robust recipe is
center=0 + whiten=0 + l2norm=1; fusion (two_stage/joint) and embed_dim (20–30)
are second-order.  Override any with --center/--whiten/--l2norm 0|1.

NOTE: the merge is NOT usually the bottleneck (scHiCluster uses the same
two_stage recipe); feature quality and the model matter more.

Usage:
    python extract_embeddings_c2.py --run_dir logs/schic_em_c2_.../ --output emb/ \
        --fusion joint --embed_dim 30
"""

import argparse
import logging
import os
from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def load_checkpoint_c2(ckpt_path: str, chrom: str):
    """Returns (Z_hats (N,M,K), b_dist (D_band,), cell_names, D_band)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    Z_hats = ckpt["Z_hats"].float()                       # (N, M, K)
    cell_names = ckpt.get("cell_names") or []
    if not cell_names:
        from src.data.schic import ScHiCDataset
        data_cfg = ckpt["config"]["data"]
        ds = ScHiCDataset(
            scool_path=data_cfg["scool_path"],
            chrom=chrom,
            min_contacts=data_cfg.get("min_contacts", 0),
        )
        cell_names = list(ds.cell_names)
        if len(cell_names) != Z_hats.shape[0]:
            raise RuntimeError(
                f"{chrom}: dataset has {len(cell_names)} cells but ckpt has "
                f"{Z_hats.shape[0]} Z_hats."
            )

    from src.models.geometric import BandBiasNetwork
    bb_cfg = ckpt["config"].get("model", {}).get("band_bias", {})
    band_bias = BandBiasNetwork(
        hidden_dim=bb_cfg.get("hidden_dim", 64),
        num_layers=bb_cfg.get("num_layers", 3),
    )
    band_bias.load_state_dict(ckpt["band_bias"])

    D_band = int(ckpt["config"].get("model", {}).get("band_width", 50))
    M = Z_hats.shape[1]
    D_band = min(D_band, M - 1)

    d_idx = torch.arange(1, D_band + 1, dtype=torch.float32)
    with torch.no_grad():
        b_dist = band_bias.forward_distances(torch.log1p(d_idx))   # (D_band,)
    return Z_hats, b_dist, cell_names, D_band


def compute_band_features_c2(
    Z_hats: torch.Tensor,         # (N, M, K)
    b_dist: torch.Tensor,         # (D_band,)
    D_band: int,
    use_softplus: bool = True,
    batch_size: int = 32,
    device: torch.device = None,
) -> np.ndarray:
    """
    For each cell, compute log1p(softplus(ZZᵀ + B(d))) on upper-tri band
    and flatten to (M * D_band,).  Returns (N, M·D_band) float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, M, K = Z_hats.shape
    X = np.zeros((N, M * D_band), dtype=np.float32)
    b_dist_dev = b_dist.to(device)
    Z_dev = Z_hats.to(device)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Z = Z_dev[start:end]
            B_sz = Z.shape[0]
            F_band = torch.zeros(B_sz, M, D_band, device=device, dtype=Z.dtype)
            for d in range(1, D_band + 1):
                if d >= M:
                    break
                inner = (Z[:, : M - d, :] * Z[:, d:, :]).sum(-1)
                F_band[:, : M - d, d - 1] = inner
            F_band = F_band + b_dist_dev.view(1, 1, D_band)
            if use_softplus:
                feat = torch.log1p(F.softplus(F_band))
            else:
                feat = torch.log1p(F_band.clamp(min=0))
            rows = torch.arange(M, device=device).view(M, 1)
            dist = torch.arange(1, D_band + 1, device=device).view(1, D_band)
            feat = feat * ((rows + dist) < M).view(1, M, D_band)
            X[start:end] = feat.reshape(B_sz, -1).cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return X


def reduce_svd(X: np.ndarray, dim: int, center: bool, whiten: bool):
    """Mean-center (PCA) → TruncatedSVD → optional singular-value whitening."""
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    dim = min(dim, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, algorithm="arpack", random_state=0)
    r = svd.fit_transform(X)
    if whiten:
        r = r / svd.singular_values_
    return r.astype(np.float32), svd


def block_gram(X: np.ndarray, center: bool, per_chrom_norm: bool) -> np.ndarray:
    """N×N contribution X_c X_cᵀ of one chrom to the joint Gram matrix."""
    Xc = X.astype(np.float64)
    if center:
        Xc -= Xc.mean(axis=0, keepdims=True)
    if per_chrom_norm:
        tr = float((Xc * Xc).sum())
        if tr > 0:
            Xc *= np.sqrt(Xc.shape[0] / tr)   # mean per-cell sq-norm → 1
    return Xc @ Xc.T


def gram_to_embedding(G: np.ndarray, embed_dim: int, whiten: bool) -> np.ndarray:
    """Top-`embed_dim` PCA scores from a centered-feature Gram matrix."""
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:embed_dim]
    lam = np.clip(w[order], 1e-12, None)
    U = V[:, order]
    return (U if whiten else U * np.sqrt(lam)).astype(np.float32)


def common_cells(feats) -> List[str]:
    """Cells present in every chromosome, in the first chrom's order.
    `feats` is a list of (chrom, X, names)."""
    common = set.intersection(*[set(n) for _, _, n in feats])
    return [n for n in feats[0][2] if n in common]


def build_embedding(feats, order, fusion, per_chrom_dim, embed_dim,
                    center, whiten, l2norm) -> np.ndarray:
    """Canonical embedding builder shared by this script and sweep_embedding.py,
    so a config swept here reproduces the deployed extraction exactly.

    feats: list of (chrom, X (N_chrom, M*D), names); order: target cell list.
    """
    if fusion == "two_stage":
        parts = []
        for _, X, names in feats:
            idx = {n: i for i, n in enumerate(names)}
            rows = [idx[n] for n in order]
            r, _ = reduce_svd(X[rows], per_chrom_dim, center, whiten)
            parts.append(r)
        emb, _ = reduce_svd(np.concatenate(parts, axis=1), embed_dim, center, whiten)
    else:  # joint
        N = len(order)
        G = np.zeros((N, N), dtype=np.float64)
        for _, X, names in feats:
            idx = {n: i for i, n in enumerate(names)}
            rows = [idx[n] for n in order]
            G += block_gram(X[rows], center, per_chrom_norm=whiten)
        emb = gram_to_embedding(G, embed_dim, whiten)
    if l2norm:
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb


def discover_chrom_ckpts(run_dir: str) -> Dict[str, str]:
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"No checkpoints/ under {run_dir}")
    result: Dict[str, str] = {}
    for entry in sorted(os.listdir(ckpt_root)):
        cand = os.path.join(ckpt_root, entry, "schic_em_c2_final.pt")
        if os.path.isfile(cand):
            result[entry] = cand
    if not result:
        raise FileNotFoundError(f"No schic_em_c2_final.pt under {ckpt_root}")
    return result


def main() -> None:
    import torch
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir")
    src.add_argument("--checkpoint")
    p.add_argument("--chrom", default=None)
    p.add_argument("--fusion", choices=["two_stage", "joint"], default="two_stage",
                   help="how to merge chromosomes (see module docstring).")
    p.add_argument("--per_chrom_dim", type=int, default=50,
                   help="[two_stage] PCA dim per chromosome before concat.")
    p.add_argument("--embed_dim", type=int, default=30,
                   help="final cell-embedding dimensionality (sweep sweet spot).")
    p.add_argument("--exclude_chroms", default="chrY,chrM,chrEBV")
    p.add_argument("--only_chroms", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--no_softplus", action="store_true")
    # 0/1 flags (sweep-winning defaults: center off, whiten off, l2norm on).
    p.add_argument("--center", type=int, choices=[0, 1], default=0,
                   help="mean-centering (PCA). Sweep winner: 0 (off, with l2norm on).")
    p.add_argument("--whiten", type=int, choices=[0, 1], default=0,
                   help="divide components by singular values. Winner: 0 "
                        "(whitening tanks silhouette).")
    p.add_argument("--l2norm", type=int, choices=[0, 1], default=1,
                   help="L2-normalize final embedding rows (cosine-like). Winner: 1.")
    p.add_argument("--output", default="./embeddings_c2")
    args = p.parse_args()
    # Guard against an empty value (e.g. an unset shell variable: --run_dir "").
    if not args.run_dir and not args.checkpoint:
        p.error("provide a non-empty --run_dir <dir> or --checkpoint <file> "
                "(an unset shell variable like --run_dir \"$RUN_DIR\" gives empty).")
    center, whiten, softplus = bool(args.center), bool(args.whiten), not args.no_softplus

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.run_dir:
        chrom_ckpts = discover_chrom_ckpts(args.run_dir)
        if args.only_chroms:
            keep = {c.strip() for c in args.only_chroms.split(",") if c.strip()}
            chrom_ckpts = {c: v for c, v in chrom_ckpts.items() if c in keep}
        else:
            excl = {c.strip() for c in args.exclude_chroms.split(",") if c.strip()}
            chrom_ckpts = {c: v for c, v in chrom_ckpts.items() if c not in excl}
    else:
        chrom = args.chrom or (torch.load(args.checkpoint, map_location="cpu")
                               ["config"]["data"].get("chrom") or "")
        chrom_ckpts = {chrom: args.checkpoint}
    if not chrom_ckpts:
        p.error("No chromosomes selected.")

    feats = []
    for chrom, path in chrom_ckpts.items():
        Z, b_dist, names, D = load_checkpoint_c2(path, chrom)
        X = compute_band_features_c2(Z, b_dist, D, use_softplus=softplus,
                                     batch_size=args.batch_size, device=device)
        feats.append((chrom, X, names))
        log.info("  %s: N=%d, feat=%d", chrom, X.shape[0], X.shape[1])

    cell_names = common_cells(feats)
    if not cell_names:
        raise RuntimeError("No cells common to all chromosomes.")
    embedding = build_embedding(feats, cell_names, args.fusion, args.per_chrom_dim,
                                args.embed_dim, center, whiten, bool(args.l2norm))

    np.save(os.path.join(args.output, "embedding.npy"), embedding)
    with open(os.path.join(args.output, "cell_names.txt"), "w") as f:
        f.write("\n".join(cell_names))
    log.info("Saved embedding %s (fusion=%s) → %s",
             embedding.shape, args.fusion, args.output)


if __name__ == "__main__":
    main()
