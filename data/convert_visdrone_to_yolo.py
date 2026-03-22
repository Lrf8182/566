from pathlib import Path
import cv2
from tqdm import tqdm

# VisDrone 类别映射：
# 原始类别通常从 1 开始，这里转成 0-based
# 忽略类别 0 和 ignore regions
VALID_CLASSES = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

def convert_one_split(images_dir, annotations_dir, labels_dir):
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

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

                    if cls_id not in VALID_CLASSES:
                        continue
                    if bw <= 0 or bh <= 0:
                        continue

                    new_cls = VALID_CLASSES[cls_id]

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

if __name__ == "__main__":
    root = Path("./data")

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

    print("Done.")