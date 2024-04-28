from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # model = YOLO("yolov8n.yaml")  # no pretrained model

    # Train the model
    results = model.train(
        data="path/to/data.yaml",
        epochs=50,
        imgsz=128,
        batch=-1,
        device="0",
        save=True,
        save_period=2,
        project="path/to/results",
        name="pico2_size128_n_pretrained",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        resume=False,
        plots=True,
        seed=314,
        patience=5,
    )
