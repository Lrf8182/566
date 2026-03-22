from pathlib import Path
import cv2
import json
from ultralytics import YOLO, RTDETR
from models.router import SimpleRouter

def extract_preds(result, conf_thres=0.25):
    boxes_xyxy = []
    scores = []
    classes = []

    if result.boxes is None:
        return boxes_xyxy, scores, classes

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    for b, s, c in zip(xyxy, conf, cls):
        if float(s) >= conf_thres:
            boxes_xyxy.append(b.tolist())
            scores.append(float(s))
            classes.append(int(c))

    return boxes_xyxy, scores, classes

def main():
    small_model = YOLO("runs/yolo11n_visdrone/weights/best.pt")
    large_model = RTDETR("runs/rtdetr_l_visdrone/weights/best.pt")
    router = SimpleRouter()

    image_dir = Path("data/VisDrone2019-DET-val/images")
    output_json = []

    for img_path in sorted(image_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        small_res = small_model.predict(
            source=str(img_path),
            imgsz=960,
            conf=0.25,
            verbose=False
        )[0]

        boxes, scores, classes = extract_preds(small_res, conf_thres=0.25)
        escalate, stats = router.decide(boxes, scores, img.shape)

        if escalate:
            final_res = large_model.predict(
                source=str(img_path),
                imgsz=960,
                conf=0.25,
                verbose=False
            )[0]
            used_model = "large"
            final_boxes, final_scores, final_classes = extract_preds(final_res, conf_thres=0.25)
        else:
            used_model = "small"
            final_boxes, final_scores, final_classes = boxes, scores, classes

        output_json.append({
            "image": img_path.name,
            "used_model": used_model,
            "router_stats": stats,
            "num_preds": len(final_scores),
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
        })

        print(f"{img_path.name}: {used_model}, stats={stats}")

    with open("runs/collab_predictions.json", "w") as f:
        json.dump(output_json, f, indent=2)

if __name__ == "__main__":
    main()