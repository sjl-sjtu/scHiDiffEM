"""
Shared down-sampling benchmark for imputation evaluation.

To compare imputation methods fairly every predictor must impute from the SAME
down-sampled data and be scored against the SAME held-out count complement.
This module builds that benchmark ONCE and persists it, so model
inference (impute_model.py) and scoring (evaluate.py --mode imputation) are fully
decoupled — you don't re-run any model each time you evaluate.

A benchmark lives in <bench_dir>/<chrom>/ and contains:
  downsampled.scool   binomially-thinned contacts (the shared model input)
  ref.npz             held-out band `ref` (N,M,D) + cell names and metadata

`build_or_load_bench` is deterministic given (scool, chrom, frac, seed, cell set),
and reuses an existing benchmark unless `replace=True`.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .schic import ScHiCDataset

log = logging.getLogger(__name__)


def split_observed_heldout_counts(cnt: np.ndarray, frac: float, rng):
    """Binomially split integer counts into observed and disjoint held-out counts."""
    cnt = np.maximum(np.asarray(cnt, dtype=np.int64), 0)
    observed = rng.binomial(cnt, frac).astype(np.int32)
    heldout = (cnt - observed).astype(np.int32)
    return observed, heldout


def valid_band_mask(M: int, D: int) -> np.ndarray:
    i = np.arange(M)[:, None]
    d = np.arange(1, D + 1)[None, :]
    return (i + d) < M



def _name_variants(name: str) -> List[str]:
    out = [name]
    parts = name.split(".")
    while len(parts) > 1:
        parts.pop()
        out.append(".".join(parts))
    return out


def _scool_name_paths(scool_path):
    """Return scool cell names/paths without opening every cell cooler."""
    import cooler.fileops

    paths = list(cooler.fileops.list_coolers(scool_path))
    return [(p.split("/")[-1], p) for p in paths]


def _normalize_whitelist(scool_path, chrom, whitelist, name_paths=None) -> Optional[set]:
    """Map a possibly suffix-stripped whitelist onto actual scool cell names."""
    if whitelist is None:
        return None
    if name_paths is None:
        name_paths = _scool_name_paths(scool_path)
    by_variant = {}
    for name, _ in name_paths:
        for v in _name_variants(name):
            by_variant.setdefault(v, []).append(name)
    resolved = set()
    missing = []
    ambiguous = []
    for name in whitelist:
        hit = None
        for v in _name_variants(str(name)):
            cand = by_variant.get(v, [])
            if len(cand) == 1:
                hit = cand[0]
                break
            if len(cand) > 1:
                ambiguous.append(str(name))
                break
        if hit is None:
            missing.append(str(name))
        else:
            resolved.add(hit)
    if missing:
        log.warning("  whitelist: %d names not found in %s; first missing: %s",
                    len(missing), chrom, missing[:3])
    if ambiguous:
        raise ValueError(
            f"Whitelist has {len(ambiguous)} ambiguous names for {chrom}; "
            f"first ambiguous: {ambiguous[:3]}"
        )
    return resolved

def _resolve_cells(scool_path, chrom, max_cells, whitelist, seed) -> List[str]:
    ds = ScHiCDataset(scool_path, chrom, cell_whitelist=whitelist)
    # Sampling indices are meaningful only after canonicalizing order. Cooler
    # group iteration and ScHiCDataset may expose the same cells differently.
    names = sorted(ds.cell_names)
    if max_cells and len(names) > max_cells:
        sel = np.random.RandomState(seed).choice(len(names), max_cells, replace=False)
        names = [names[i] for i in sorted(sel)]
    return names


def build_or_load_bench(
    scool_path: str,
    chrom: str,
    bench_dir: str,
    frac: float = 0.5,
    seed: int = 0,
    D_band: int = 50,
    max_cells: int = 0,
    cell_whitelist: Optional[set] = None,
    replace: bool = False,
    need_ref: bool = True,
) -> Dict:
    """Return a dict {ref, scool_path, cell_names, cell_paths, M, D, frac, seed}.

    For frac < 1, ``ref`` is the held-out count complement (original minus
    thinned input), never the original map containing the observed input.
    Builds and persists the down-sampled scool and reference band if missing.
    """
    import cooler
    import cooler.fileops
    import pandas as pd

    cdir = os.path.join(bench_dir, chrom)
    scool_out = os.path.join(cdir, "downsampled.scool")
    ref_path = os.path.join(cdir, "ref.npz")
    meta_path = os.path.join(cdir, "meta.json")

    # Listing HDF5 group paths is cheap. With a training whitelist, full-frac
    # imputation can return before ScHiCDataset fetches chr1 for every cell.
    name_paths = _scool_name_paths(scool_path)
    cell_whitelist = _normalize_whitelist(
        scool_path, chrom, cell_whitelist, name_paths=name_paths,
    )
    # Select the benchmark cohort before constructing ScHiCDataset. Its
    # chromosome validation touches every admitted cell, so passing the full
    # training whitelist and truncating afterwards made a 200-cell benchmark
    # scan thousands of cells on every new cache build.
    if max_cells:
        available = sorted(
            n for n, _ in name_paths
            if cell_whitelist is None or n in cell_whitelist
        )
    else:
        available = []
    if max_cells and len(available) > max_cells:
        sel = np.random.RandomState(seed).choice(
            len(available), max_cells, replace=False,
        )
        available = [available[i] for i in sorted(sel)]
    if max_cells:
        cell_whitelist = set(available)
    # A persisted full-frac reference is the source of truth for benchmark cell
    # identity. Reuse it during model inference instead of independently drawing
    # another seeded subset from a potentially different source ordering.
    if (frac >= 1.0 - 1e-12 and not replace
            and os.path.isfile(ref_path) and os.path.isfile(meta_path)):
        meta = json.load(open(meta_path))
        with np.load(ref_path, allow_pickle=True) as z:
            cached_names = z["cell_names"].astype(str).tolist()
            cached_ref = z["ref"].astype(np.float32) if need_ref else None
        path_by_name = dict(name_paths)
        expected_names = sorted(
            cell_whitelist if cell_whitelist is not None else set(path_by_name)
        )
        reusable = (
            meta.get("frac") == frac
            and meta.get("seed") == seed
            and int(meta.get("max_cells", 0)) == int(max_cells)
            and int(meta.get("D", -1)) == min(D_band, int(meta.get("M", 0)) - 1)
            and cached_names == expected_names
            and all(n in path_by_name for n in cached_names)
        )
        if reusable:
            log.info(
                "  %s: using %d cells from persisted full-frac benchmark",
                chrom, len(cached_names),
            )
            return {
                "ref": cached_ref,
                "scool_path": scool_path,
                "cell_names": cached_names,
                "cell_paths": [path_by_name[n] for n in cached_names],
                "M": int(meta["M"]),
                "D": int(meta["D"]),
                "frac": frac,
                "seed": seed,
            }
    if frac >= 1.0 - 1e-12 and not need_ref and cell_whitelist is not None:
        selected = sorted(
            ((n, p) for n, p in name_paths if n in cell_whitelist),
            key=lambda item: item[0],
        )
        if max_cells and len(selected) > max_cells:
            sel = np.random.RandomState(seed).choice(
                len(selected), max_cells, replace=False,
            )
            selected = [selected[i] for i in sorted(sel)]
        if not selected:
            raise ValueError(f"No whitelist cells found in {scool_path}")
        cell_names = [n for n, _ in selected]
        cell_paths = [p for _, p in selected]
        c0 = cooler.Cooler(f"{scool_path}::{cell_paths[0]}")
        M_req = len(c0.bins().fetch(chrom))
        if M_req <= 0:
            raise ValueError(f"Chromosome {chrom} has no bins in {cell_names[0]}")
        D_req = min(D_band, M_req - 1)
        log.info("  %s: frac=1; using original scool directly (%d cells)",
                 chrom, len(cell_names))
        return {"ref": None, "scool_path": scool_path,
                "cell_names": cell_names, "cell_paths": cell_paths,
                "M": M_req, "D": D_req, "frac": frac, "seed": seed}

    # For a persisted downsample benchmark, validate and return it before
    # ScHiCDataset performs an expensive chromosome fetch for every cell.
    if frac < 1.0 - 1e-12 and cell_whitelist is not None:
        selected = sorted(
            ((n, p) for n, p in name_paths if n in cell_whitelist),
            key=lambda item: item[0],
        )
        if max_cells and len(selected) > max_cells:
            sel = np.random.RandomState(seed).choice(
                len(selected), max_cells, replace=False,
            )
            selected = [selected[i] for i in sorted(sel)]
        if not selected:
            raise ValueError(f"No whitelist cells found in {scool_path}")
        fast_names = [n for n, _ in selected]
        c0 = cooler.Cooler(f"{scool_path}::{selected[0][1]}")
        fast_M = len(c0.bins().fetch(chrom))
        fast_D = min(D_band, fast_M - 1)
        cache_ready = (
            not replace
            and os.path.isfile(scool_out)
            and os.path.isfile(meta_path)
            and (not need_ref or os.path.isfile(ref_path))
        )
        if cache_ready:
            meta = json.load(open(meta_path))
            cached_names = None
            z = None
            if os.path.isfile(ref_path):
                z = np.load(ref_path, allow_pickle=True)
                cached_names = z["cell_names"].astype(str).tolist()
            elif "cell_names" in meta:
                cached_names = list(meta["cell_names"])
            reusable = (
                meta.get("frac") == frac
                and meta.get("seed") == seed
                and meta.get("reference_kind") == "heldout_complement_v1"
                and int(meta.get("M", -1)) == fast_M
                and int(meta.get("D", -1)) == fast_D
                and cached_names == fast_names
            )
            if reusable:
                paths = {p.split("/")[-1]: p
                         for p in cooler.fileops.list_coolers(scool_out)}
                if all(n in paths for n in fast_names):
                    ref = z["ref"].astype(np.float32) if need_ref else None
                    if z is not None:
                        z.close()
                    log.info("  %s: reusing benchmark without scanning source cells (%d cells)",
                             chrom, len(fast_names))
                    return {"ref": ref, "scool_path": scool_out,
                            "cell_names": fast_names,
                            "cell_paths": [paths[n] for n in fast_names],
                            "M": fast_M, "D": fast_D,
                            "frac": frac, "seed": seed}
            if z is not None:
                z.close()

    ds = ScHiCDataset(scool_path, chrom, cell_whitelist=cell_whitelist)
    cell_names = sorted(ds.cell_names)
    if max_cells and len(cell_names) > max_cells:
        sel = np.random.RandomState(seed).choice(
            len(cell_names), max_cells, replace=False,
        )
        cell_names = [cell_names[i] for i in sorted(sel)]
    M_req = ds.M
    D_req = min(D_band, M_req - 1)

    if frac >= 1.0 - 1e-12:
        name2path = {n: p for n, p in zip(ds.cell_names, ds.cell_paths)}
        cell_paths = [name2path[n] for n in cell_names]
        if not need_ref:
            log.info("  %s: frac=1; using original scool directly (%d cells)",
                     chrom, len(cell_names))
            return {"ref": None, "scool_path": scool_path,
                    "cell_names": cell_names, "cell_paths": cell_paths,
                    "M": M_req, "D": D_req, "frac": frac, "seed": seed}

        if (not replace and os.path.isfile(ref_path) and os.path.isfile(meta_path)):
            meta = json.load(open(meta_path))
            with np.load(ref_path, allow_pickle=True) as z:
                names = z["cell_names"].astype(str).tolist()
                cached_ref = z["ref"].astype(np.float32)
            expected_n = min(max_cells, len(ds.cell_names)) if max_cells else len(ds.cell_names)
            reusable = (
                meta.get("frac") == frac
                and meta.get("seed") == seed
                and int(meta.get("M", -1)) == M_req
                and int(meta.get("D", -1)) == D_req
                and int(meta.get("max_cells", 0)) == int(max_cells)
                and len(names) == expected_n
                and len(set(names)) == len(names)
                and all(n in name2path for n in names)
            )
            if reusable:
                log.info("  %s: reusing full-frac reference (%d cells)", chrom, len(names))
                return {"ref": cached_ref, "scool_path": scool_path,
                        "cell_names": names,
                        "cell_paths": [name2path[n] for n in names],
                        "M": int(meta["M"]), "D": int(meta["D"]),
                        "frac": frac, "seed": seed}
            log.info("  %s: full-frac reference cache cell set/params changed; rebuilding.", chrom)

        os.makedirs(cdir, exist_ok=True)
        name2idx = {n: i for i, n in enumerate(ds.cell_names)}
        M, D = M_req, D_req
        N = len(cell_names)
        ref = np.zeros((N, M, D), dtype=np.float32)
        for n, name in enumerate(cell_names):
            coo = ds[name2idx[name]]["Y_coo"].numpy()
            i = coo[:, 0].astype(np.int64)
            d = (coo[:, 1] - coo[:, 0]).astype(np.int64)
            keep = (d >= 1) & (d <= D) & (i < M)
            ref[n, i[keep], d[keep] - 1] = coo[:, 2][keep].astype(np.float32)
        np.savez_compressed(ref_path, ref=ref, cell_names=np.array(cell_names, str))
        json.dump({"frac": frac, "seed": seed, "M": M, "D": D, "max_cells": max_cells}, open(meta_path, "w"))
        log.info("  %s: built full-frac reference (%d cells, M=%d, D=%d) -> %s",
                 chrom, N, M, D, cdir)
        return {"ref": ref, "scool_path": scool_path, "cell_names": cell_names,
                "cell_paths": cell_paths, "M": M, "D": D, "frac": frac, "seed": seed}
    if (not replace and os.path.isfile(scool_out) and os.path.isfile(ref_path)
            and os.path.isfile(meta_path)):
        meta = json.load(open(meta_path))
        z = np.load(ref_path, allow_pickle=True)
        names = z["cell_names"].astype(str).tolist()
        reusable = (
            meta.get("frac") == frac
            and meta.get("seed") == seed
            and meta.get("reference_kind") == "heldout_complement_v1"
            and int(meta.get("M", -1)) == M_req
            and int(meta.get("D", -1)) == D_req
            and names == cell_names
        )
        if reusable:
            paths = {p.split("/")[-1]: p for p in cooler.fileops.list_coolers(scool_out)}
            if all(n in paths for n in names):
                log.info("  %s: reusing benchmark (%d cells)", chrom, len(names))
                return {"ref": z["ref"].astype(np.float32), "scool_path": scool_out,
                        "cell_names": names, "cell_paths": [paths[n] for n in names],
                        "M": int(meta["M"]), "D": int(meta["D"]),
                        "frac": frac, "seed": seed}
        log.info("  %s: benchmark cache cell set/params changed; rebuilding.", chrom)

    os.makedirs(cdir, exist_ok=True)
    name2idx = {n: i for i, n in enumerate(ds.cell_names)}
    c0 = cooler.Cooler(f"{scool_path}::{ds.cell_paths[0]}")
    M = ds.M
    D = min(D_band, M - 1)
    bins = c0.bins().fetch(chrom)[["chrom", "start", "end"]].reset_index(drop=True)
    rng = np.random.RandomState(seed)

    N = len(cell_names)
    ref = np.zeros((N, M, D), dtype=np.float32)
    cell_pixels: OrderedDict = OrderedDict()
    for n, name in enumerate(cell_names):
        coo = ds[name2idx[name]]["Y_coo"].numpy()
        i = coo[:, 0].astype(np.int64)
        j = coo[:, 1].astype(np.int64)
        cnt = coo[:, 2].astype(np.int64)
        d = j - i
        kb = (d >= 1) & (d <= D) & (i < M)
        thinned, heldout = split_observed_heldout_counts(cnt, frac, rng)
        ref[n, i[kb], d[kb] - 1] = heldout[kb].astype(np.float32)
        nz = thinned > 0
        cell_pixels[name] = pd.DataFrame({
            "bin1_id": i[nz].astype(np.int32),
            "bin2_id": j[nz].astype(np.int32),
            "count": thinned[nz],
        }).sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)

    if os.path.exists(scool_out):
        os.remove(scool_out)
    cooler.create_scool(scool_out, bins=bins, cell_name_pixels_dict=cell_pixels,
                        ordered=True, dupcheck=False)
    np.savez_compressed(ref_path, ref=ref, cell_names=np.array(cell_names, str))
    json.dump({"frac": frac, "seed": seed, "M": M, "D": D,
               "max_cells": max_cells,
               "reference_kind": "heldout_complement_v1"},
              open(meta_path, "w"))
    paths = {p.split("/")[-1]: p for p in cooler.fileops.list_coolers(scool_out)}
    log.info("  %s: built benchmark (%d cells, M=%d, D=%d) → %s", chrom, N, M, D, cdir)
    return {"ref": ref, "scool_path": scool_out, "cell_names": cell_names,
            "cell_paths": [paths[n] for n in cell_names], "M": M, "D": D,
            "frac": frac, "seed": seed}
