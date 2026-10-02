"""
scHi-C dataset loader for .scool format.

A .scool file is an HDF5 file where each cell stores its own contact matrix
in cooler format. We use the `cooler` library to read these files.

Expected .scool structure:
  /cells/<cell_name>/bins        (chromosome, start, end, weight)
  /cells/<cell_name>/pixels      (bin1_id, bin2_id, count)
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import scipy.sparse as sp

try:
    import cooler
    import cooler.fileops
except ImportError:
    raise ImportError(
        "cooler is required for scHiC data loading. "
        "Install via: pip install cooler"
    )


def safe_fetch_cooler(c: cooler.Cooler, chrom: str) -> sp.spmatrix:
    """Safely fetch a sparse matrix for a chromosome from a Cooler object,
    falling back to pixel reconstruction if the native fetch fails due to bugs in large/sparse chromosomes.
    """
    try:
        # 1. 正常情况下依然尝试最快的原生读取
        return c.matrix(balance=False, sparse=True).fetch(chrom)
    except Exception:
        # 2. 如果踩到 Bug 崩溃了，立刻切换到 pixels 模式安全重建
        try:
            # 获取该染色体在全局 bin 中的偏移量 and bin 总数
            chrom_offset = c.offset(chrom)
            chrom_extent = c.extent(chrom)
            N = chrom_extent[1] - chrom_extent[0] # 该染色体应有的 bin 数量
            
            # 直接拉取 1D 的 pixels（这一步绝对不会报 Index 错误）
            pixels = c.pixels().fetch(chrom)
            
            if len(pixels) == 0:
                # 如果这细胞在这条染色体上真的惨到连 1 个 read 都没有，直接给全 0 矩阵
                mat = sp.csr_matrix((N, N))
            else:
                # 将全局 bin_id 转换为染色体局部相对坐标
                row = pixels['bin1_id'].values - chrom_offset
                col = pixels['bin2_id'].values - chrom_offset
                data = pixels['count'].values
                
                # 构建上三角稀疏矩阵并补全对称矩阵
                mat = sp.csr_matrix((data, (row, col)), shape=(N, N))
                mat = mat + mat.T - sp.diags(mat.diagonal())
            return mat
        except Exception as e:
            raise e


_MAX_CELL_TRIES = 20  # how many cells to try before giving up on a chrom/binsize read


def confirm_chrom_binsize(scool_path: str, cell_paths: List[str], chrom: str) -> Optional[int]:
    """Confirm this chromosome's OWN bin width directly from its bin table.

    This does not trust cooler.binsize as ground truth for every chromosome —
    a global attribute can be stale/wrong for a chrom that was rebuilt or
    patched separately, and that mismatch is exactly what causes bp positions
    to land outside chrom.sizes downstream (Higashi's 'row index exceeds
    matrix dimensions'; similar range bugs in any consumer that turns bin
    index -> bp via a resolution).

    Uses the MAJORITY bin width, not strict uniformity: real cooler bin
    tables routinely have a handful of irregular-width bins around assembly
    gaps/centromeres/chrom ends even at a genuinely fixed resolution, so
    requiring every bin to match is too strict and false-negatives on normal
    data.

    `cell_paths` is tried in order (not just the first) — some cells in a real
    scool can have a corrupted/unreadable cooler group (ScHiCDataset already
    tolerates this per-cell via safe_fetch_cooler / its own try-and-skip loop),
    so hard-coding a single cell means one broken cell fails EVERY chromosome
    even though most other cells would confirm fine.

    Returns the majority width in bp, or None if it cannot be confirmed from
    any of the tried cells (missing/empty bin table on all of them, or no
    width appears often enough to call it the resolution) — callers should
    SKIP the chromosome rather than guess.
    """
    for cell_path in cell_paths[:_MAX_CELL_TRIES]:
        try:
            c = cooler.Cooler(f"{scool_path}::{cell_path}")
            bins = c.bins().fetch(chrom)
        except Exception:
            continue
        if bins is None or len(bins) < 1:
            continue
        widths = (bins["end"] - bins["start"]).to_numpy()
        widths = widths[widths > 0]
        if len(widths) == 0:
            continue
        if len(widths) == 1:
            return int(widths[0])
        vals, counts = np.unique(widths, return_counts=True)
        top = counts.argmax()
        if counts[top] / len(widths) < 0.5:   # no clear majority: try the next cell
            continue
        return int(vals[top])
    return None


def _stored_binsize(scool_path: str, cell_paths: List[str]) -> Optional[int]:
    """cooler.binsize from the first cell that can actually be opened — cheap
    HDF5 attr read, no pixel data touched. Tries multiple cells in case the
    first one has a corrupted cooler group (see confirm_chrom_binsize)."""
    for cell_path in cell_paths[:_MAX_CELL_TRIES]:
        try:
            c = cooler.Cooler(f"{scool_path}::{cell_path}")
            bs = c.binsize
        except Exception:
            continue
        return int(bs) if bs is not None else None
    return None


def resolve_resolution_rough(
    scool_path: str,
    chroms: List[str],
    fallback_resolution: Optional[int] = None,
) -> int:
    """Cheap, approximate resolution for consumers that don't actually need bp
    accuracy: our own EM training and scHiCluster RWR only use `resolution` to
    convert RWR's `output_dist` cutoff (default 500,000,000 bp) into a bin
    count, and that cutoff is already far larger than any real chromosome, so
    it never binds — resolution just needs to be roughly right (not off by
    orders of magnitude), not exactly confirmed.

    Priority: cooler.binsize (single cheap read) if set, else the first bin's
    width of the first chromosome (no majority-vote, no per-chrom dropping —
    unlike resolve_chroms_confirmed, being off by a few bp here changes
    nothing downstream), else fallback_resolution. Tries multiple cells (not
    just the first) for both, since a corrupted first cell would otherwise
    fail resolution for every chromosome even though other cells are fine.
    """
    paths = cooler.fileops.list_coolers(scool_path)
    if not paths:
        raise RuntimeError(f"No cells in {scool_path}")
    if not chroms:
        raise RuntimeError("No chromosomes available for resolution inference")

    bs = _stored_binsize(scool_path, paths)
    if bs is not None:
        return bs

    for cell_path in paths[:_MAX_CELL_TRIES]:
        try:
            c = cooler.Cooler(f"{scool_path}::{cell_path}")
            bins = c.bins().fetch(chroms[0])
            if bins is not None and len(bins) > 0:
                w = int(bins["end"].iloc[0] - bins["start"].iloc[0])
                if w > 0:
                    return w
        except Exception:
            continue

    if fallback_resolution is not None:
        return int(fallback_resolution)
    raise RuntimeError(
        f"Cannot resolve even a rough resolution for {scool_path}: "
        "cooler.binsize is None and bin-table inference failed on every cell "
        f"tried (up to {_MAX_CELL_TRIES}). Set data.resolution in the config "
        "or pass --resolution."
    )


def resolve_chroms_confirmed(
    scool_path: str,
    chroms: List[str],
    fallback_resolution: Optional[int] = None,
) -> Tuple[List[str], int]:
    """Per-chromosome-confirmed resolution — for consumers that genuinely need
    an exact bp resolution: Higashi/Fast-Higashi (data.txt needs real bp
    coordinates) and scVI-3D (its --max_dist cutoff, unlike RWR's, is small
    enough to actually bind). For anything insensitive to small resolution
    error, use the cheaper resolve_resolution_rough instead.

    For each chrom, confirm its OWN bin width (confirm_chrom_binsize). Chroms
    that can't be confirmed are DROPPED (logged), not guessed at or allowed to
    fail the whole run. Among confirmed chroms, the majority resolution wins;
    any chrom disagreeing with the majority is also dropped (a single global
    `resolution` is what every downstream consumer — Higashi's config, our own
    bp-position export — assumes, so a minority-resolution chrom is not usable
    in the same run, but that's no reason to abort the rest).

    Priority: cooler.binsize (the file's own stored metadata, read once from a
    single cell) wins outright and is used for every chrom with no further
    checking — it's authoritative when present. The per-chromosome
    confirm-or-skip dance below only runs when cooler.binsize is None.

    Returns (kept_chroms, resolution). Raises if nothing could be confirmed
    and no fallback_resolution is given.
    """
    logger = logging.getLogger(__name__)
    paths = cooler.fileops.list_coolers(scool_path)
    if not paths:
        raise RuntimeError(f"No cells in {scool_path}")

    bs = _stored_binsize(scool_path, paths)
    if bs is not None:
        return list(chroms), bs

    per_chrom: Dict[str, int] = {}
    for ch in chroms:
        res = confirm_chrom_binsize(scool_path, paths, ch)
        if res is None:
            logger.warning("  %s: could not confirm bin width from its own bin "
                            "table — skipping.", ch)
            continue
        per_chrom[ch] = res

    if not per_chrom:
        if fallback_resolution is not None:
            logger.warning("No chromosome's bin width could be confirmed; using "
                           "fallback_resolution=%d for all.", fallback_resolution)
            return list(chroms), int(fallback_resolution)
        raise RuntimeError(
            f"Could not confirm bin width for ANY chromosome in {scool_path} "
            "from their own bin tables. Pass an explicit resolution."
        )

    counts: Dict[int, int] = {}
    for res in per_chrom.values():
        counts[res] = counts.get(res, 0) + 1
    majority_res = max(counts, key=counts.get)

    kept = [ch for ch in chroms if per_chrom.get(ch) == majority_res]
    dropped = [ch for ch in chroms if ch in per_chrom and per_chrom[ch] != majority_res]
    if dropped:
        logger.warning("  Dropping chromosomes whose confirmed bin width disagrees "
                        "with the majority (%d bp): %s",
                        majority_res, {ch: per_chrom[ch] for ch in dropped})
    return kept, majority_res


class ScHiCDataset(Dataset):
    """
    Dataset for single-cell Hi-C contact maps stored in .scool format.

    Each item is the upper-triangle of one cell's contact matrix for the
    specified chromosome, returned in sparse (COO) form alongside library
    size and number of bins.
    """

    def __init__(
        self,
        scool_path: str,
        chrom: str,
        min_contacts: int = 0,
        cell_whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        """
        cell_whitelist: optional iterable of cell *names* (basename of the
        scool internal path) to use.  When supplied, cells outside the list
        are skipped immediately, regardless of `min_contacts`.  This is the
        recommended way to keep the same cell set across chromosomes 鈥?        compute it once via `src.data.schic_filter.select_cells_by_total_contacts`.
        Per-chrom `min_contacts` is still applied on top (set to 0 when you
        already have a global whitelist).
        """
        super().__init__()
        self.scool_path = scool_path
        self.chrom = chrom
        self.min_contacts = min_contacts
        whitelist = set(cell_whitelist) if cell_whitelist is not None else None

        # Enumerate cells
        all_cell_paths = cooler.fileops.list_coolers(scool_path)
        self.cell_paths: List[str] = []
        self.cell_names: List[str] = []

        for cp in all_cell_paths:
            cell_name = cp.split("/")[-1]
            if whitelist is not None and cell_name not in whitelist:
                continue
            c = cooler.Cooler(f"{scool_path}::{cp}")
            if chrom not in c.chromnames:
                continue
            try:
                bins_chrom = c.bins().fetch(chrom)
            except (IndexError, ValueError):
                continue
            if len(bins_chrom) == 0:
                continue
            # If the bin table is valid, a fetch with no contacts should return
            # an empty matrix. IndexError here means this cell/chrom is
            # structurally unusable, so skip it even when min_contacts=0.
            try:
                pixels_chrom = safe_fetch_cooler(c, chrom)
                total = int(pixels_chrom.data.sum())
            except Exception as e:
                import traceback
                logger.warning(f"Error loading {cp} on {chrom}: {e}")
                traceback.print_exc()
                continue
            if total < min_contacts:
                continue
            self.cell_paths.append(cp)
            self.cell_names.append(cell_name)

        if len(self.cell_paths) == 0:
            raise ValueError(
                f"No cells found in {scool_path} with chromosome {chrom} "
                f"and >= {min_contacts} contacts."
            )

        # Compute M (number of bins for this chrom) from first valid cell
        c0 = cooler.Cooler(f"{scool_path}::{self.cell_paths[0]}")
        bins = c0.bins().fetch(chrom)
        self.M: int = len(bins)

    def __len__(self) -> int:
        return len(self.cell_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Y_coo: LongTensor (nnz, 3) of (i, j, count), i <= j, 0-indexed within chrom
            s: float 鈥?library size (sum of upper-triangle counts)
            M: int 鈥?number of bins
            cell_name: str
        """
        c = cooler.Cooler(f"{self.scool_path}::{self.cell_paths[idx]}")

        # Fetch pixels restricted to this chromosome
        pixels = safe_fetch_cooler(c, self.chrom)
        # pixels is scipy COO matrix (M x M)
        coo = pixels.tocoo()

        # Keep strict upper triangle only (i < j): excludes diagonal
        # (diagonal = d=0 self-ligation artifacts; band model only uses d>=1)
        mask = coo.row < coo.col
        rows = coo.row[mask].astype(np.int64)
        cols = coo.col[mask].astype(np.int64)
        data = coo.data[mask].astype(np.float32)

        s = float(data.sum())
        nnz = len(data)

        Y_coo = torch.zeros((nnz, 3), dtype=torch.float32)
        Y_coo[:, 0] = torch.from_numpy(rows).float()
        Y_coo[:, 1] = torch.from_numpy(cols).float()
        Y_coo[:, 2] = torch.from_numpy(data)

        return {
            "Y_coo": Y_coo,       # (nnz, 3): i, j, count
            "s": s,
            "M": self.M,
            "cell_name": self.cell_names[idx],
            "idx": idx,
        }

    def get_dense(self, idx: int) -> Tuple[torch.Tensor, float]:
        """
        Returns the full symmetric dense contact matrix and library size.
        Shape: (M, M) float32.  Used for SVD initialization.
        """
        c = cooler.Cooler(f"{self.scool_path}::{self.cell_paths[idx]}")
        mat = c.matrix(balance=False, sparse=False).fetch(self.chrom)
        Y = torch.from_numpy(mat.astype(np.float32))
        # Symmetrize (should already be symmetric but enforce)
        Y = 0.5 * (Y + Y.t())
        s = float(Y.triu().sum().item())
        return Y, s

    def get_all_dense(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load all cells into a dense tensor. May be memory-intensive for large M.
        Returns:
            Y_all: (N, M, M) float32
            s_all: (N,) float32
        """
        N = len(self)
        Y_all = torch.zeros((N, self.M, self.M), dtype=torch.float32)
        s_all = torch.zeros(N, dtype=torch.float32)
        for i in range(N):
            Y, s = self.get_dense(i)
            Y_all[i] = Y
            s_all[i] = s
        return Y_all, s_all


def _collate_sparse(batch: List[Dict]) -> Dict:
    """
    Collate a batch of sparse scHiC samples.
    Pads Y_coo tensors to the same length.
    """
    max_nnz = max(item["Y_coo"].shape[0] for item in batch)
    B = len(batch)
    M = batch[0]["M"]

    Y_padded = torch.zeros((B, max_nnz, 3), dtype=torch.float32)
    nnz_counts = torch.zeros(B, dtype=torch.long)
    s_tensor = torch.zeros(B, dtype=torch.float32)
    idxs = []
    cell_names = []

    for b, item in enumerate(batch):
        nnz = item["Y_coo"].shape[0]
        Y_padded[b, :nnz] = item["Y_coo"]
        nnz_counts[b] = nnz
        s_tensor[b] = item["s"]
        idxs.append(item["idx"])
        cell_names.append(item["cell_name"])

    return {
        "Y_padded": Y_padded,   # (B, max_nnz, 3)
        "nnz_counts": nnz_counts,
        "s": s_tensor,
        "M": M,
        "idxs": idxs,
        "cell_names": cell_names,
    }


def sparse_to_dense(Y_coo: torch.Tensor, M: int) -> torch.Tensor:
    """
    Convert a single cell's COO tensor (nnz, 3) to dense symmetric (M, M).
    """
    Y = torch.zeros((M, M), dtype=torch.float32)
    i = Y_coo[:, 0].long()
    j = Y_coo[:, 1].long()
    v = Y_coo[:, 2]
    Y[i, j] = v
    Y[j, i] = v  # symmetrize
    return Y


def create_schic_dataloader(
    scool_path: str,
    chrom: str,
    batch_size: int = 1,
    min_contacts: int = 0,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[ScHiCDataset, DataLoader]:
    dataset = ScHiCDataset(scool_path, chrom, min_contacts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_sparse,
    )
    return dataset, loader


def create_schic_dataloader_from_config(config: dict) -> Tuple[ScHiCDataset, DataLoader]:
    d = config["data"]
    return create_schic_dataloader(
        scool_path=d["scool_path"],
        chrom=d["chrom"],
        batch_size=d.get("batch_size", 1),
        min_contacts=d.get("min_contacts", 0),
        shuffle=d.get("shuffle", False),
        num_workers=d.get("num_workers", 0),
    )


def resolve_chroms(scool_path: str, chrom_spec) -> List[str]:
    """
    Resolve a chrom specification to a concrete list of chromosome names.

    chrom_spec can be:
      "all"          鈫?every chrom present in the .scool file
      "auto"         鈫?autosomes only (excludes chrX, chrY, chrM, chrEBV, 鈥?
      "chr1"         鈫?single chrom
      ["chr1","chr2"]鈫?explicit list (also accepts comma-separated string)
    """
    # Get all chroms from the first cell
    all_cell_paths = cooler.fileops.list_coolers(scool_path)
    if not all_cell_paths:
        raise ValueError(f"No cells found in {scool_path}")
    c0 = cooler.Cooler(f"{scool_path}::{all_cell_paths[0]}")
    available = list(c0.chromnames)

    if isinstance(chrom_spec, list):
        return [c for c in chrom_spec if c in available]

    if isinstance(chrom_spec, str) and "," in chrom_spec:
        return [c.strip() for c in chrom_spec.split(",") if c.strip() in available]

    if chrom_spec == "all":
        return available

    if chrom_spec == "auto":
        exclude = {"chrX", "chrY", "chrM", "chrEBV", "chrUn"}
        return [
            c for c in available
            if c not in exclude and not any(c.startswith(p) for p in ("chrUn", "random", "alt"))
        ]

    # Single chromosome name
    if chrom_spec in available:
        return [chrom_spec]

    raise ValueError(f"Chromosome '{chrom_spec}' not found in {scool_path}. "
                     f"Available: {available}")
