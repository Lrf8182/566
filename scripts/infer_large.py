import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR


DEFAULT_CONFIG = {
    "model": "rtdetr-l.pt",
    "source": "data/VisDrone2019-DET-val-third/images",
    "imgsz": 960,
    "conf": 0.25,
    "device": "0",
    "project": "runs/predict",
    "name": "rtdetr_l_pretrained_visdrone",
    "save_txt": False,
    "save_conf": False,
    "max_images": None,
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


def build_parser():
    parser = argparse.ArgumentParser(description="Run RT-DETR large inference without training first.")
    parser.add_argument("--config", type=Path, help="Optional YAML config path for inference arguments.")
    parser.add_argument("--model", help="Model weights path, or an Ultralytics model name such as rtdetr-l.pt.")
    parser.add_argument("--source", help="Image file, directory, or glob for inference.")
    parser.add_argument("--imgsz", type=int, help="Inference image size.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument("--device", help="CUDA device, for example 0 or cpu.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Run name.")
    parser.add_argument("--save-txt", dest="save_txt", type=str2bool, help="Whether to save YOLO txt outputs.")
    parser.add_argument("--save-conf", dest="save_conf", type=str2bool, help="Whether to save confidences in txt.")
    parser.add_argument("--max-images", dest="max_images", type=int, help="Only run on the first N images.")
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


def resolve_source(source, max_images):
    source_path = Path(source)
    if max_images is None or not source_path.is_dir():
        return source

    image_paths = sorted(
        p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_path}")
    return [str(p) for p in image_paths[:max_images]]


def main():
    args = parse_args()
    source = resolve_source(args.source, args.max_images)

    model = RTDETR(args.model)
    common_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "exist_ok": True,
        "verbose": True,
    }

    if isinstance(source, list):
        for image_path in source:
            model.predict(source=image_path, **common_kwargs)
    else:
        model.predict(source=source, **common_kwargs)


if __name__ == "__main__":
    main()
