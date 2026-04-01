from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data=r"D:\runway_model\data\data.yaml",  # path to dataset YAML
        epochs=75,  # number of training epochs
        imgsz=640,  # training image size
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # # Perform object detection on an image
    # results = model(r"D:\runway_model\data\train\images\Oil-1251-_jpg.rf.e8f50642b00bda74cfde6d2d28dbad68.jpg")
    # results[0].show()

    # # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
