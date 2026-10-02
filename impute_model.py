"""
Produce a c2-model imputation on the shared down-sampling benchmark.

This is the MODEL side of the decoupled imputation benchmark: it reads the
persisted down-sampled scool built by src.data.downsample (the same input every
predictor sees), re-infers Z cold-start from the model's training preprocessing
of that thinned data (no leakage), and writes the imputed band to a `.npz` that
`evaluate.py --mode imputation --pred name=<file>` then scores.

Decoupling models from evaluation this way means you run the model once and
evaluate as many times as you like without re-running it.

Usage:
    # 1) build/refresh the benchmark and 2) impute with the model:
    python impute_model.py --checkpoint logs/.../chr1/schic_em_c2_final.pt \
        --chrom chr1 --scool data/x.scool --bench_dir bench/ \
        --down_frac 0.5 --output preds/model_chr1.npz
"""

import argparse
import logging
import os

import numpy as np
import torch

from src.data import preprocess as P
from src.data.downsample import build_or_load_bench, valid_band_mask

log = logging.getLogger(__name__)

def load_cell_whitelist(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        names = {line.strip() for line in f if line.strip()}
    log.info("Loaded cell whitelist: %d cells from %s", len(names), path)
    return names


def build_em(ckpt_path, M, K, D, device):
    """Rebuild the EM object matching how the checkpoint was TRAINED.  The score
    net is built CONDITIONAL or UNCONDITIONAL to match the saved weights (detected
    from the state_dict), and config.training.method is resolved consistently so
    inference uses the same method.  Robust to old checkpoints lacking the tag."""
    from src.models.geometric import BandBiasNetwork
    from src.methods.schic_em_c2 import ScHiCEMC2
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt["config"]
    is_cond = any(("cond_encoder" in k) or ("null_cond" in k)
                  for k in ckpt["score_net"].keys())
    # method: explicit config > legacy ablation tag > infer from net type.
    method = config.get("training", {}).get("method")
    if method is None:
        method = ("dps" if config.get("model", {}).get("ablation") == "dps"
                  else ("cfg" if is_cond else "map"))
    config.setdefault("training", {})["method"] = method
    if is_cond:
        from src.models.score_net_cond import create_cond_score_net
        net = create_cond_score_net(M, K, D, config)
    else:
        from src.models.score_net import create_score_net
        net = create_score_net("resnet1d", M, K, config)
    bb_cfg = config.get("model", {}).get("band_bias", {})
    bb = BandBiasNetwork(hidden_dim=bb_cfg.get("hidden_dim", 64),
                         num_layers=bb_cfg.get("num_layers", 3))
    em = ScHiCEMC2(net, bb, config, device)
    em.load_checkpoint(ckpt_path)
    em.D_band = D
    em._valid_mask = torch.from_numpy(valid_band_mask(M, D)).to(device)
    didx = torch.arange(1, D + 1, device=device, dtype=torch.float32)
    short_range_boost = 1.0 + 3.0 * torch.exp(-didx / 8.0)
    em._dist_w = (1.0 / (M - didx).clamp(min=1.0)) * short_range_boost
    em._refresh_b_dist()
    em.score_net.eval()
    return em, config


def _dense_pred(F_band: torch.Tensor) -> torch.Tensor:
    """Clean NB mean mapped to log1p scale for evaluate.py alignment."""
    import torch.nn.functional as Fn
    mu = Fn.softplus(F_band)
    return torch.log1p(mu)


def _name_variants(name):
    out = [str(name)]
    parts = str(name).split(".")
    while len(parts) > 1:
        parts.pop()
        out.append(".".join(parts))
    return out


def predict_checkpoint_z(em, names, M, batch_size, min_cutoff=0.01):
    """Reconstruct full-fraction training cells from checkpoint Z_hats."""
    ckpt_names = list(em._cell_names)
    if em.Z_hats is None or not ckpt_names:
        raise ValueError("Checkpoint does not contain aligned Z_hats/cell_names")
    if len(ckpt_names) != em.Z_hats.shape[0]:
        raise ValueError("Checkpoint Z_hats and cell_names have different lengths")

    aliases = {}
    ambiguous = set()
    for i, name in enumerate(ckpt_names):
        for key in _name_variants(name):
            if key in aliases and aliases[key] != i:
                ambiguous.add(key)
            else:
                aliases[key] = i
    for key in ambiguous:
        aliases.pop(key, None)

    source_idx = []
    missing = []
    for name in names:
        hit = next((aliases[v] for v in _name_variants(name) if v in aliases), None)
        if hit is None:
            missing.append(name)
        else:
            source_idx.append(hit)
    if missing:
        raise ValueError(
            f"Checkpoint Z alignment is missing {len(missing)}/{len(names)} cells; "
            f"first missing: {missing[:3]}"
        )

    N, D = len(names), em.D_band
    preds = np.zeros((N, M, D), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            idx = torch.as_tensor(source_idx[s:e], device=em.device, dtype=torch.long)
            out = _dense_pred(em.reconstruct_band(em.Z_hats[idx])).cpu().numpy()
            if min_cutoff > 0:
                out = np.maximum(0.0, out - min_cutoff)
            preds[s:e] = out
            if e == N or s == 0 or e % max(batch_size, N // 5) < batch_size:
                log.info("  checkpoint reconstruction %d/%d", e, N)
    return preds


def reinfer_map(em, y_raw, exposure, M, infer_steps, batch_size, device,
                min_cutoff=0.01, x_aux=None) -> np.ndarray:
    import math
    N, _, D = y_raw.shape
    K = em.K
    if infer_steps is not None:
        em.K_E = int(infer_steps)
    preds = np.zeros((N, M, D), dtype=np.float32)
    valid = em._valid_mask.view(1, M, D).float()
    dist_w = em._dist_w.view(1, 1, D)
    scale = 1.0 / math.sqrt(K)
    for s in range(0, N, batch_size):
        idx = list(range(s, min(s + batch_size, N)))
        yb = torch.from_numpy(y_raw[idx]).to(device)
        eb = torch.from_numpy(exposure[idx]).to(device)
        w = dist_w * valid
        population_cond = None
        if x_aux is not None and em._cluster_feature_centers is not None:
            if em.population_conditioned:
                Z0, population_cond = em.initialize_Z_from_aux(
                    x_aux[idx], return_population=True,
                )
            else:
                Z0 = em.initialize_Z_from_aux(x_aux[idx])
        elif em._seed_Z is not None:
            seed_idx = torch.randint(em._seed_Z.shape[0], (len(idx),), device=device)
            Z0 = em._seed_Z[seed_idx].clone()
            if em.seed_noise > 0:
                Z0.add_(em.seed_noise * torch.randn_like(Z0))
        else:
            Z0 = scale * torch.randn(len(idx), M, K, device=device, dtype=torch.float32)
        if em.map_prior_mode == "denoise":
            lam = em.prior_weight
        else:
            Zc = Z0.detach().clone().requires_grad_(True)
            dl = em._raw_nb_nll(em.reconstruct_band(Zc), yb, eb, w, len(idx))
            dg = torch.autograd.grad(dl, Zc)[0]
            sc = em._prior_score(Zc.detach(), population_cond)
            zn = dg.flatten(1).norm(dim=1).mean().item()
            pn = sc.flatten(1).norm(dim=1).mean().item()
            lam = em.prior_weight * (zn / pn if pn > 1e-8 else 1.0)
        Z = em.e_step_map_batch(
            Z0, yb, eb, M, lam, population_cond=population_cond,
        )
        with torch.no_grad():
            out = _dense_pred(em.reconstruct_band(Z)).cpu().numpy()
            if min_cutoff > 0:
                out = np.maximum(0.0, out - min_cutoff)
            preds[idx] = out
        log.info("  re-infer %d/%d (lambda=%.3g, K_E=%d)",
                 min(s + batch_size, N), N, lam, em.K_E)
    return preds

def sample_pred(em, x_aux, y_raw, exposure, M, device, min_cutoff=0.0) -> np.ndarray:
    """Reverse-diffusion posterior prediction per cell via em.sample_posterior
    (CFG or DPS depending on the checkpoint's trained method)."""
    N, _, D = x_aux.shape
    preds = np.zeros((N, M, D), dtype=np.float32)
    for n in range(N):
        Z = em.sample_posterior(
            torch.from_numpy(x_aux[n]).to(device),
            y_raw=torch.from_numpy(y_raw[n]).to(device),
            exposure=torch.from_numpy(exposure[n]).to(device),
            num_samples=1,
        ).to(device)
        with torch.no_grad():
            out = _dense_pred(em.reconstruct_band(Z)).cpu().numpy()[0:1]
        if min_cutoff > 0:
            out = np.maximum(0.0, out - min_cutoff)
        preds[n] = out[0]
    return preds

def run_impute(checkpoint, chrom, scool, bench_dir, output, down_frac=0.5, seed=0,
               max_cells=0, infer_steps=None, method=None, batch_size=64,
               device=None, dps_guidance_scale=None, dps_num_steps=None,
               min_cutoff=0.01, replace=False, cell_whitelist=None,
               force_reinfer=False) -> str:
    """Impute one (checkpoint, chrom) on the shared benchmark -> write `output`
    (.npz with band + cell_names).  Returns the output path.

    Skips re-inference if `output` already exists, unless `replace=True` --
    matches run_schicluster_impute.py's --replace convention."""
    if os.path.isfile(output) and not replace:
        log.info("  %s: cached prediction at %s -- skip (pass --replace to force).",
                 chrom, output)
        return output
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = torch.load(checkpoint, map_location="cpu")["config"]
    D_band = int(config.get("model", {}).get("band_width", 50))
    K = int(config.get("model", {}).get("K"))
    # Reproduce the checkpoint's auxiliary preprocessing (method + params, e.g.
    # smooth_sigma). Raw counts/exposure below define the inference likelihood.
    pp = config.get("training", {}).get("preprocess", {}) or {}
    train_method = pp.get("method", "rwr")
    cfg = {"n_jobs": 4,
           **(config.get("training", {}).get("rwr_target", {}) or {}),
           **pp}

    log.info("Benchmark (%s, frac=%.2f) ...", chrom, down_frac)
    bench = build_or_load_bench(scool, chrom, bench_dir, frac=down_frac, seed=seed,
                                D_band=D_band, max_cells=max_cells,
                                cell_whitelist=load_cell_whitelist(cell_whitelist),
                                need_ref=False)
    M, D, names = bench["M"], bench["D"], bench["cell_names"]

    em, _ = build_em(checkpoint, M, K, D, device)
    log.info(
        "  checkpoint=%s | MAP prior=%s/%s | population_cond=%s | "
        "prior_weight=%.3g | z_scale=%.4g",
        os.path.abspath(checkpoint), em.map_prior_mode,
        em.map_denoise_scaling, em.population_conditioned,
        em.prior_weight, em.z_scale,
    )

    # Full-fraction MAP on the same training cells already has converged Z_hats
    # in the checkpoint. Reconstruct them directly instead of reading every
    # cooler and repeating hundreds of random-start E-steps.
    if method is None:
        method = em.method
    if (down_frac >= 1.0 - 1e-12 and method == "map" and not force_reinfer):
        log.info("  full-frac MAP: reconstructing checkpoint Z_hats directly ...")
        pred = predict_checkpoint_z(
            em, names, M, batch_size, min_cutoff=min_cutoff,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        log.info("  saving uncompressed prediction (fast I/O) ...")
        np.savez(
            output, band=pred, cell_names=np.array(names, str),
            value_space=np.array("log1p"),
            train_population_size=np.array(len(em._cell_names), dtype=np.int64),
            checkpoint=np.array(os.path.abspath(checkpoint)),
            map_prior_mode=np.array(em.map_prior_mode),
            map_denoise_scaling=np.array(em.map_denoise_scaling),
            population_conditioned=np.array(em.population_conditioned),
        )
        log.info("Saved model imputation %s -> %s", pred.shape, output)
        return output

    log.info("  loading down-sampled raw counts for EM likelihood ...")
    raw = P.load_raw_band(
        bench["scool_path"], bench["cell_paths"], chrom, M, D,
        progress_prefix="    ",
    )
    exposure = em._compute_exposure_from_raw(raw, em.exposure_max, em.exposure_min)

    x_obs = None
    need_seed_aux = (method == "map" and em._cluster_feature_centers is not None)
    if method != "map" or need_seed_aux:
        log.info("  building auxiliary input via '%s' ...", train_method)
        if train_method in ("rwr", "schicluster"):
            from src.data.schic import resolve_resolution_rough
            res = resolve_resolution_rough(bench["scool_path"], [chrom])
            x_obs = P.compute_observation_band(
                "rwr", bench["scool_path"], bench["cell_paths"],
                chrom, res, M, D, cfg, "    ",
            )
        elif train_method == "raw_log1p":
            x_obs = np.log1p(raw).astype(np.float32)
        else:
            x = np.log1p(P.apply_bandnorm(raw))
            if train_method == "bandnorm_smooth":
                x = P.gaussian_smooth_i(x, float(cfg.get("smooth_sigma", 1.0)))
            x_obs = x.astype(np.float32)

    # Default inference was resolved before the full-fraction fast path.
    if method == "cfg" and not em.conditional:
        raise ValueError("--method cfg needs a cfg-trained (conditional) checkpoint; "
                         "this one was trained with method=%s." % em.method)
    if method == "dps":                              # optional DPS guidance overrides
        if dps_guidance_scale is not None:
            em.dps_guidance_scale = float(dps_guidance_scale)
        if dps_num_steps is not None:
            em.sample_steps = int(dps_num_steps)

    log.info("  re-inferring (method=%s, trained=%s) ...", method, em.method)
    if method == "map":                              # cold-start MAP E-step
        pred = reinfer_map(em, raw, exposure, M, infer_steps, batch_size, device,
                           min_cutoff=min_cutoff, x_aux=x_obs)
    else:                                            # dps/cfg reverse sampler
        pred = sample_pred(
            em, x_obs, raw, exposure, M, device, min_cutoff=min_cutoff,
        )


    zero_frac = float((pred <= 0).mean())
    pos = pred[pred > 0]
    log.info("  prediction zeros=%.1f%%, mean(pos)=%.4g",
             100.0 * zero_frac, float(pos.mean()) if pos.size else 0.0)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    log.info("  saving uncompressed prediction (fast I/O) ...")
    np.savez(
        output, band=pred, cell_names=np.array(names, str),
        value_space=np.array("log1p"),
        train_population_size=np.array(len(em._cell_names), dtype=np.int64),
        checkpoint=np.array(os.path.abspath(checkpoint)),
        map_prior_mode=np.array(em.map_prior_mode),
        map_denoise_scaling=np.array(em.map_denoise_scaling),
        population_conditioned=np.array(em.population_conditioned),
    )
    log.info("Saved model imputation %s -> %s", pred.shape, output)
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--scool", required=True)
    p.add_argument("--bench_dir", required=True,
                   help="shared benchmark dir (same as evaluate.py --bench_dir).")
    p.add_argument("--down_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cells", type=int, default=0)
    p.add_argument("--cell_whitelist", default=None,
                   help="Optional file with one cell name per line; constrains the shared benchmark cell set.")
    p.add_argument("--infer_steps", type=int, default=None,
                   help="MAP updates; default uses training K_E from the checkpoint.")
    p.add_argument("--method", choices=["map", "cfg", "dps"], default=None,
                   help="inference method; default = the checkpoint's TRAINED method "
                        "(consistency). map = cold-start MAP; cfg/dps = reverse sampler "
                        "(cfg needs a cfg-trained checkpoint).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--dps_guidance_scale", type=float, default=None,
                   help="[dps] override likelihood-guidance strength (default 1.0 is "
                        "standard for normalized likelihood gradient).")
    p.add_argument("--dps_num_steps", type=int, default=None,
                   help="[dps] override reverse-diffusion steps.")
    p.add_argument("--min_cutoff", type=float, default=0.01,
                   help="Background intensity cutoff threshold to zero out weak noise (default 0.01).")
    p.add_argument("--output", required=True, help="output .npz (band + cell_names).")
    p.add_argument("--replace", action="store_true",
                   help="recompute even if --output already exists (default: skip cached).")
    p.add_argument("--force_reinfer", action="store_true",
                   help="At down_frac=1, ignore checkpoint Z_hats and run cold-start inference.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(args.device) if args.device else None
    run_impute(args.checkpoint, args.chrom, args.scool, args.bench_dir, args.output,
               dps_guidance_scale=args.dps_guidance_scale, dps_num_steps=args.dps_num_steps,
               min_cutoff=args.min_cutoff,
               down_frac=args.down_frac, seed=args.seed, max_cells=args.max_cells,
               cell_whitelist=args.cell_whitelist,
               infer_steps=args.infer_steps, method=args.method,
               batch_size=args.batch_size, device=device, replace=args.replace,
               force_reinfer=args.force_reinfer)


if __name__ == "__main__":
    main()
