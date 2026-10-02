"""
Global cell-quality filter for scHi-C.

Avoids the per-chrom intersection problem: a high-depth cell can have <N
contacts on a small chromosome simply because the chromosome is small.
Filtering per-chrom + taking cross-chrom intersection erases such cells.
The standard practice (Tan2021, Higashi, scHiCluster) is to filter by
**total contacts across autosomes** once, globally, then use the same
cell list for every chromosome.

Public API:
  - compute_cell_total_contacts(scool_path, ...) → {cell_name: int}
  - select_cells_by_total_contacts(scool_path, min_total_contacts, ...) →
      list of cell names passing the threshold.  Supports JSON cache so
      repeated runs skip the scool scan.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Iterable, List, Optional

import cooler
import cooler.fileops

logger = logging.getLogger(__name__)


def _cell_name_from_path(cp: str) -> str:
    return cp.split("/")[-1]


def compute_cell_total_contacts(
    scool_path: str,
    exclude_chroms: Iterable[str] = ("chrY", "chrM", "chrEBV"),
    use_info_nnz: bool = False,
) -> Dict[str, int]:
    """Return {cell_name: total_contact_count}.

    `use_info_nnz=True` (legacy, fast): reads cooler.info['nnz'].
        Counts all pixels in the cell (intra-chrom + inter-chrom + excluded
        chroms), so it's a slight overestimate.  Sufficient for relative
        ranking and global thresholding.

    `use_info_nnz=False` (default): reads cooler.info['sum'], the raw contact
        count total. If an older cooler lacks that field, falls back to stored
        cis counts across non-excluded chromosomes. This sequencing-depth
        measure is much less sensitive to bin resolution than nnz.
    """
    excl = set(exclude_chroms)
    cell_paths = cooler.fileops.list_coolers(scool_path)
    out: Dict[str, int] = {}
    for cp in cell_paths:
        name = _cell_name_from_path(cp)
        c = cooler.Cooler(f"{scool_path}::{cp}")
        if use_info_nnz:
            n = int(c.info.get("nnz", 0))
        else:
            info_sum = c.info.get("sum")
            if info_sum is not None:
                out[name] = int(round(float(info_sum)))
                continue
            n = 0
            for ch in c.chromnames:
                if ch in excl:
                    continue
                try:
                    from .schic import safe_fetch_cooler
                    mat = safe_fetch_cooler(c, ch)
                    # Cooler expands symmetric-upper storage to a symmetric
                    # matrix. Off-diagonal contacts therefore occur twice,
                    # while diagonal contacts occur once.
                    total = float(mat.sum())
                    diagonal = float(mat.diagonal().sum())
                    n += int(round((total + diagonal) / 2.0))
                except Exception:
                    # Cell has zero pixels on this chrom — cooler raises or safe_fetch_cooler fails.
                    pass
        out[name] = n
    return out


def select_cells_by_total_contacts(
    scool_path: str,
    min_total_contacts: int,
    exclude_chroms: Iterable[str] = ("chrY", "chrM", "chrEBV"),
    cache_path: Optional[str] = None,
    use_info_nnz: bool = False,
) -> List[str]:
    """Return cell names whose total contacts ≥ min_total_contacts.

    `cache_path`: optional JSON path.  Saves the per-cell totals dict so
    rerunning on the same scool skips the (potentially slow) scan.  Cache
    is keyed by scool_path + use_info_nnz so changes invalidate it.
    """
    excluded = tuple(exclude_chroms)
    cache_key = {
        "scool_path": scool_path,
        "use_info_nnz": use_info_nnz,
        "exclude_chroms": sorted(set(excluded)),
    }

    totals: Optional[Dict[str, int]] = None
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if all(cached.get(k) == v for k, v in cache_key.items()):
                totals = {k: int(v) for k, v in cached["totals"].items()}
                logger.info(
                    "Loaded cell-totals cache from %s (%d cells)",
                    cache_path, len(totals),
                )
            else:
                logger.info(
                    "Cache at %s is stale (scool/options changed); recomputing.",
                    cache_path,
                )
        except (json.JSONDecodeError, KeyError):
            logger.info("Cache at %s is malformed; recomputing.", cache_path)

    if totals is None:
        logger.info(
            "Scanning %s for per-cell total contacts (use_info_nnz=%s) ...",
            scool_path, use_info_nnz,
        )
        totals = compute_cell_total_contacts(
            scool_path,
            exclude_chroms=excluded,
            use_info_nnz=use_info_nnz,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            payload = dict(cache_key)
            payload["totals"] = totals
            with open(cache_path, "w") as f:
                json.dump(payload, f)
            logger.info("Saved cell-totals cache to %s", cache_path)

    kept = [name for name, n in totals.items() if n >= min_total_contacts]
    if totals:
        sorted_vals = sorted(totals.values())
        median = sorted_vals[len(sorted_vals) // 2]
        p10 = sorted_vals[len(sorted_vals) // 10]
        p90 = sorted_vals[(len(sorted_vals) * 9) // 10]
    else:
        median = p10 = p90 = 0
    logger.info(
        "Global filter: kept %d / %d cells | min_total_contacts=%d | "
        "totals p10=%d, median=%d, p90=%d",
        len(kept), len(totals), min_total_contacts, p10, median, p90,
    )
    return kept
