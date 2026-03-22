import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


DEFAULT_CONFIG = {
    "model": "yolo11n.pt",
    "data": "data/visdrone.yaml",
    "epochs": 200,
    "imgsz": 960,
    "batch": 32,
    "device": "0",
    "workers": 8,
    "project": "runs",
    "name": "yolo11n_visdrone",
    "resume": False,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on VisDrone with a standard single training run."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config path for training arguments.")
    parser.add_argument("--model", help="Starting weights for the first training stage.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, help="Total epochs to reach.")
    parser.add_argument("--imgsz", type=int, help="Training image size.")
    parser.add_argument("--batch", type=int, help="Batch size.")
    parser.add_argument("--device", help="CUDA device, for example 0 or 0,1.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--project", help="Ultralytics project directory.")
    parser.add_argument("--name", help="Run name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a checkpoint and continue until --epochs is reached.",
    )
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


def validate_resume_checkpoint(args):
    if not args.resume:
        return

    model_path = Path(args.model)
    if model_path.suffix != ".pt" or not model_path.exists():
        return

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if ckpt.get("epoch", -1) < 0:
        raise ValueError(
            f"{model_path} is a completed/stripped checkpoint and cannot be resumed. "
            "Use it as the starting weights for a new fine-tuning run instead, for example: "
            f"'python scripts/train_small.py --model {model_path} --epochs 50 --name yolo11n_visdrone_plus50'."
        )


def train(args):
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=True,
        cache=False,
        degrees=0.0,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        exist_ok=True,
    )


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    validate_resume_checkpoint(args)
    train(args)


if __name__ == "__main__":
    main()
