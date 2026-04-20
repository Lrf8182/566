import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

# VisDrone 原始 10 类合并为 5 类：
# 0: person         <- pedestrian, people
# 1: two_wheeler    <- bicycle, motor
# 2: car            <- car
# 3: large_vehicle  <- van, truck, bus
# 4: tricycle       <- tricycle, awning-tricycle
# 忽略类别 0 和 ignore regions
RAW_VISDRONE_TO_MERGED = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 1,   # bicycle -> two_wheeler
    4: 2,   # car -> car
    5: 3,   # van -> large_vehicle
    6: 3,   # truck -> large_vehicle
    7: 4,   # tricycle -> tricycle
    8: 4,   # awning-tricycle -> tricycle
    9: 3,   # bus -> large_vehicle
    10: 1,  # motor -> two_wheeler
}

OLD_YOLO_TO_MERGED = {
    0: 0,  # pedestrian -> person
    1: 0,  # people -> person
    2: 1,  # bicycle -> two_wheeler
    3: 2,  # car -> car
    4: 3,  # van -> large_vehicle
    5: 3,  # truck -> large_vehicle
    6: 4,  # tricycle -> tricycle
    7: 4,  # awning-tricycle -> tricycle
    8: 3,  # bus -> large_vehicle
    9: 1,  # motor -> two_wheeler
}


def convert_one_split(images_dir, annotations_dir, labels_dir):
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total_boxes = 0
    non_empty_labels = 0

    for img_path in tqdm(image_files, desc=f"Converting {images_dir.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        ann_path = annotations_dir / f"{img_path.stem}.txt"
        out_path = labels_dir / f"{img_path.stem}.txt"
        yolo_lines = []

        if ann_path.exists():
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue

                    x, y, bw, bh = map(float, parts[:4])
                    score = int(parts[4])      # 可见性/有效性字段，部分版本可不用
                    cls_id = int(parts[5])
                    trunc = int(parts[6])
                    occ = int(parts[7])

                    if cls_id not in RAW_VISDRONE_TO_MERGED:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue

                    new_cls = RAW_VISDRONE_TO_MERGED[cls_id]

                    xc = (x + bw / 2) / w
                    yc = (y + bh / 2) / h
                    nw = bw / w
                    nh = bh / h

                    xc = min(max(xc, 0.0), 1.0)
                    yc = min(max(yc, 0.0), 1.0)
                    nw = min(max(nw, 1e-6), 1.0)
                    nh = min(max(nh, 1e-6), 1.0)

                    yolo_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))

        if yolo_lines:
            non_empty_labels += 1
            total_boxes += len(yolo_lines)

    print(
        f"{images_dir.parent.name}: {len(image_files)} images, "
        f"{non_empty_labels} non-empty labels, {total_boxes} boxes"
    )


def remap_yolo_labels_dir(labels_dir):
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        print(f"Skip remap, directory does not exist: {labels_dir}")
        return

    txt_files = sorted(labels_dir.glob("*.txt"))
    total_files = 0
    total_boxes = 0

    for label_path in tqdm(txt_files, desc=f"Remapping {labels_dir.name}"):
        remapped_lines = []

        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                if cls_id not in OLD_YOLO_TO_MERGED:
                    raise ValueError(f"Unsupported old YOLO class id {cls_id} in {label_path}")

                parts[0] = str(OLD_YOLO_TO_MERGED[cls_id])
                remapped_lines.append(" ".join(parts))

        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(remapped_lines))

        total_files += 1
        total_boxes += len(remapped_lines)

    print(f"{labels_dir}: remapped {total_files} label files, {total_boxes} boxes")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone DET annotations to merged 5-class YOLO labels."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Dataset root directory containing VisDrone2019-DET-* folders. Defaults to <repo>/data.",
    )
    parser.add_argument(
        "--remap-old-yolo-label-dir",
        dest="remap_old_yolo_label_dirs",
        action="append",
        type=Path,
        default=[],
        help="Optional existing YOLO label directory to remap in place from the old 10-class ids to the merged 5-class ids. Repeat this flag for multiple directories.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    root = args.root.resolve()

    convert_one_split(
        root / "VisDrone2019-DET-train/images",
        root / "VisDrone2019-DET-train/annotations",
        root / "VisDrone2019-DET-train/labels",
    )

    convert_one_split(
        root / "VisDrone2019-DET-val/images",
        root / "VisDrone2019-DET-val/annotations",
        root / "VisDrone2019-DET-val/labels",
    )

    for labels_dir in args.remap_old_yolo_label_dirs:
        remap_yolo_labels_dir(labels_dir.resolve())

    print("Done.")
