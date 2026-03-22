import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


DEFAULT_CONFIG = {
    "model": "yolo11n.pt",
    "data": "data/visdrone_1of3.yaml",
    "split": "val",
    "imgsz": 960,
    "batch": 8,
    "device": "0",
    "workers": 4,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "project": "runs/val",
    "name": "yolo11n_visdrone_val1of3",
    "save_json": False,
    "plots": True,
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
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO small model on the VisDrone val t, using official pretrained weights by default."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config path for evaluation arguments.")
    parser.add_argument("--model", help="Model weights or checkpoint path. Defaults to official yolo11n.pt.")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--split", help="Dataset split to evaluate, for example val or test.")
    parser.add_argument("--imgsz", type=int, help="Validation image size.")
    parser.add_argument("--batch", type=int, help="Validation batch size.")
    parser.add_argument("--device", help="CUDA device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, help="Confidence threshold used during validation.")
    parser.add_argument("--iou", type=float, help="IoU threshold used during validation/NMS.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Maximum detections per image.")
    parser.add_argument("--project", help="Output project directory.")
    parser.add_argument("--name", help="Output run name.")
    parser.add_argument("--save-json", dest="save_json", type=str2bool, help="Whether to save JSON predictions.")
    parser.add_argument("--plots", type=str2bool, help="Whether to save validation plots.")
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


def normalize_names(names):
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    if isinstance(names, list):
        return names
    raise ValueError(f"Unsupported names format: {type(names).__name__}")


def parse_args():
    parser = build_parser()
    args, _ = parser.parse_known_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_yaml_config(args.config))

    parser.set_defaults(**config)
    return parser.parse_args()


def resolve_model_path(args):
    model_path = Path(args.model)
    if model_path.exists():
        return str(model_path)
    return args.model


def load_dataset_names(data_path):
    path = Path(data_path)
    if path.suffix not in {".yaml", ".yml"} or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f) or {}

    names = data_config.get("names")
    if names is None:
        return None
    return normalize_names(names)


def ensure_class_alignment(model, data_path):
    dataset_names = load_dataset_names(data_path)
    if dataset_names is None:
        return

    model_names = normalize_names(model.names)
    if model_names == dataset_names:
        return

    raise ValueError(
        "Model classes do not match dataset classes, so validation metrics would be misleading. "
        f"Model names start with: {model_names[:10]}. "
        f"Dataset names are: {dataset_names}. "
        "For example, official yolo11n.pt uses COCO class ids where class 4 is 'airplane', "
        "but VisDrone class 4 is 'van'. "
        "Use a VisDrone-finetuned checkpoint for eval, or run inference only if you want a qualitative baseline."
    )


def print_metrics(metrics):
    if not hasattr(metrics, "box"):
        return

    box = metrics.box
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")


def main():
    args = parse_args()
    model_path = resolve_model_path(args)

    print(f"Using weights: {model_path}")
    print(f"Evaluating data={args.data}, split={args.split}")

    model = YOLO(model_path)
    ensure_class_alignment(model, args.data)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        plots=args.plots,
        exist_ok=True,
        verbose=True,
    )

    print_metrics(metrics)
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
