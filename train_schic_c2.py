"""
Train the C2 latent-diffusion EM (improved over C1).

Uses geometric-Z + diffusion EM with an exposure-aware raw-count NB likelihood:
  - Z normalization (diffusion runs on Z̃ = Z / z_scale)
  - cosine noise schedule + Min-SNR-γ DSM weighting
  - periodic GPA gauge alignment
  - unsupervised mini-pseudo-bulk clean-prior bootstrap

The configured method is shared by training and inference; the full-map config uses
Adam MAP with an unconditional seed-pretrained prior.

Usage:
    python train_schic_c2.py --config configs/schic_c2.yaml --scool data/x.scool
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

from src.data.schic import ScHiCDataset, resolve_chroms
from src.models.geometric import BandBiasNetwork
from src.models.score_net_cond import create_cond_score_net
from src.methods.schic_em_c2 import ScHiCEMC2
from src.utils.logging_utils import setup_logger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_dotted_override(config: dict, dotted_key: str, raw_value: str) -> None:
    """Set config[a][b][...][z] = yaml.safe_load(raw_value), creating dicts as needed.

    Lets --set training.prior_weight=0 reach any scalar in the config without a
    dedicated CLI flag per parameter; yaml.safe_load gives int/float/bool/str coercion
    matching how the value would parse if written directly in the yaml file.
    """
    keys = dotted_key.split(".")
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def merge_cli(config: dict, args: argparse.Namespace) -> dict:
    if args.scool:
        config.setdefault("data", {})["scool_path"] = args.scool
    if args.chrom:
        config.setdefault("data", {})["chrom"] = args.chrom
    if args.K is not None:
        config.setdefault("model", {})["K"] = args.K
    if args.em_iters is not None:
        config.setdefault("training", {})["em_iterations"] = args.em_iters
    if args.resolution is not None:
        config.setdefault("data", {})["resolution"] = args.resolution
    if args.exclude_chroms:
        config.setdefault("data", {})["exclude_chroms"] = [
            c.strip() for c in args.exclude_chroms.split(",") if c.strip()
        ]
    if args.wandb:
        config.setdefault("logging", {})["wandb"] = True
    for item in args.set or []:
        if "=" not in item:
            raise ValueError(f"--set expects key.path=value, got: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        apply_dotted_override(config, dotted_key.strip(), raw_value.strip())
    return config


def resolve_band_width(M: int, config: dict) -> int:
    D = config.get("model", {}).get("band_width") or 100
    return max(1, min(int(D), M - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scool", default=None)
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--em_iters", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None,
                        help="fixed bin size in bp; used when cooler.binsize is missing")
    parser.add_argument("--exclude_chroms", default=None,
                        help="comma-separated chromosomes to skip after resolving --chrom")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--set", action="append", default=None, metavar="KEY.PATH=VALUE",
                        help="generic config override, dotted-path into the yaml config, "
                             "e.g. --set training.prior_weight=0 --set model.K=20; "
                             "repeatable. Value is parsed with yaml.safe_load so ints/"
                             "floats/bools come through typed.")
    parser.add_argument("--method", choices=["map", "dps", "cfg"], default=None,
                        help="UNIFIED method for E-step + inference (consistency): "
                             "'map' (MAP/point, unconditional net), 'dps' "
                             "(posterior-sampling + likelihood guidance, unconditional "
                             "net), 'cfg' (posterior-sampling + classifier-free "
                             "guidance, conditional net). Default: config training.method "
                             "or 'map'.")
    args = parser.parse_args()

    config = merge_cli(load_config(args.config), args)
    seed = int(config.get("training", {}).get(
        "random_seed",
        config.get("training", {}).get("clean_prior_init", {}).get(
            "random_seed", 0),
    ))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Resolve the single source of truth for the method (CLI > config > 'map').
    method = args.method or config.get("training", {}).get("method", "map")
    config.setdefault("training", {})["method"] = method

    log_dir = config.get("logging", {}).get("dir", "./logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"schic_em_c2_{method}_{timestamp}"
    checkpoint_dir = os.path.join(log_dir, run_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger(
        log_dir=os.path.join(log_dir, run_name),
        name="train_schic_c2",
        log_file="train.log",
    )
    logging.getLogger().setLevel(logging.INFO)
    logger.info("=" * 60)
    logger.info("scHi-C EM C2 (geometric-Z + EM, improved)")
    logger.info("=" * 60)
    run_dir = os.path.join(log_dir, run_name)
    logger.info("Run dir: %s", run_dir)

    # Save the merged config (CLI overrides applied) + invocation for provenance,
    # so the run's exact settings are inspectable without opening a checkpoint.
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(run_dir, "args.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    logger.info("Saved config → %s", os.path.join(run_dir, "config.yaml"))

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)
    logger.info("Random seed: %d", seed)

    data_cfg = config["data"]
    chroms = resolve_chroms(data_cfg["scool_path"], data_cfg.get("chrom", "auto"))
    exclude_chroms = set(data_cfg.get("exclude_chroms", []) or [])
    if exclude_chroms:
        chroms = [c for c in chroms if c not in exclude_chroms]
        logger.info("Excluded chromosomes from data.exclude_chroms: %s", sorted(exclude_chroms))
    logger.info("Chromosomes (%d): %s", len(chroms), chroms)

    cell_whitelist = None
    if int(data_cfg.get("min_total_contacts", 0)) > 0:
        from src.data.schic_filter import select_cells_by_total_contacts
        cell_whitelist = select_cells_by_total_contacts(
            scool_path=data_cfg["scool_path"],
            min_total_contacts=int(data_cfg["min_total_contacts"]),
            exclude_chroms=tuple(
                data_cfg.get("exclude_chroms_for_filter", ("chrY", "chrM", "chrEBV"))
            ),
            cache_path=data_cfg.get("cell_total_cache"),
            use_info_nnz=bool(data_cfg.get("use_info_nnz_filter", True)),
        )

    K = config["model"]["K"]
    bb_cfg = config["model"].get("band_bias", {})

    wandb_run = None
    if config.get("logging", {}).get("wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project="schic-latent-diffusion-c2", name=run_name, config=config,
            )
        except ImportError:
            logger.warning("wandb not installed; W&B logging disabled.")

    for chrom in chroms:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  Chromosome: %s", chrom)
        logger.info("=" * 60)

        chrom_ckpt_dir = os.path.join(checkpoint_dir, chrom)
        os.makedirs(chrom_ckpt_dir, exist_ok=True)

        try:
            dataset = ScHiCDataset(
                scool_path=data_cfg["scool_path"],
                chrom=chrom,
                min_contacts=data_cfg.get("min_contacts", 0),
                cell_whitelist=cell_whitelist,
            )
        except ValueError as e:
            logger.warning("Skipping %s: %s", chrom, e)
            continue

        M = dataset.M
        D_cond = resolve_band_width(M, config)
        logger.info("Dataset: N=%d cells, M=%d bins, K=%d, D_cond=%d",
                    len(dataset), M, K, D_cond)

        # cfg → conditional score net; map/dps → unconditional score net.
        if method == "cfg":
            score_net = create_cond_score_net(M, K, D_cond, config)
        else:
            from src.models.score_net import create_score_net
            score_net = create_score_net("resnet1d", M, K, config)
        band_bias = BandBiasNetwork(
            hidden_dim=bb_cfg.get("hidden_dim", 64),
            num_layers=bb_cfg.get("num_layers", 3),
        )
        em = ScHiCEMC2(
            score_net=score_net,
            band_bias=band_bias,
            config=config,
            device=device,
        )

        if args.resume and len(chroms) == 1:
            logger.info("Resuming from %s", args.resume)
            em.load_checkpoint(args.resume)

        em.em_train(dataset, checkpoint_dir=chrom_ckpt_dir, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("All chromosomes complete. Results in: %s",
                os.path.join(log_dir, run_name))


if __name__ == "__main__":
    main()
