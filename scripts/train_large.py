from ultralytics import RTDETR

def main():
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data="data/visdrone.yaml",
        epochs=100,
        imgsz=960,
        batch=8,
        device=0,
        workers=8,
        project="runs",
        name="rtdetr_l_visdrone",
        pretrained=True,
        cache=False,
    )

if __name__ == "__main__":
    main()