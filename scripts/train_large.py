import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR


DEFAULT_CONFIG = {
    "model": "rtdetr-l.pt",
    "data": "data/visdrone.yaml",
    "epochs": 100,
    "imgsz": 960,
    "batch": 8,
    "device": "0",
    "workers": 8,
    "project": "runs",
    "name": "rtdetr_l_visdrone",
    "cache": False,
    "amp": True,
    "pretrained": True,
    "deterministic": True,
    "resume": False,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_cache(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    if lowered in {"ram", "disk"}:
        return lowered
    raise argparse.ArgumentTypeError("Cache must be one of true/false/ram/disk.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR-L on VisDrone with configurable single-GPU or multi-GPU settings."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config path for training arguments.")
    parser.add_argument("--model", help="Starting weights, for example rtdetr-l.pt or a checkpoint path.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, help="Training image size.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument(
        "--device",
        help="CUDA device string, for example 0, 0,1 or 0,1,2,3 for DDP. Use cpu for CPU-only runs.",
    )
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--project", help="Ultralytics project directory.")
    parser.add_argument("--name", help="Run name.")
    parser.add_argument("--cache", type=parse_cache, help="Dataset cache mode: false, true, ram or disk.")
    parser.add_argument("--amp", type=str2bool, help="Enable automatic mixed precision.")
    parser.add_argument("--pretrained", type=str2bool, help="Whether to use pretrained weights.")
    parser.add_argument("--deterministic", type=str2bool, help="Enable deterministic training for reproducibility.")
    parser.add_argument("--resume", action="store_true", help="Resume from a checkpoint.")
    return parser


def load_yaml_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level.")

    unknown_keys = sorted(set(config) - set(DEFAULT_CONFIG))
    if unknown_keys:
        raise ValueError(
            f"Unsupported config keys in {config_path}: {', '.join(unknown_keys)}. "
            f"Supported keys: {', '.join(DEFAULT_CONFIG)}."
        )

    return config


def parse_args():
    parser = build_parser()
    args, _ = parser.parse_known_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_yaml_config(args.config))

    parser.set_defaults(**config)
    return parser.parse_args()


def train(args):
    model = RTDETR(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=args.cache,
        amp=args.amp,
        pretrained=args.pretrained,
        deterministic=args.deterministic,
        resume=args.resume,
        exist_ok=True,
    )


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    train(args)


if __name__ == "__main__":
    main()
