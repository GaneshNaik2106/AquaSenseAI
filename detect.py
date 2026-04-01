from ultralytics import YOLO

# Load your trained model (best.pt after training)
model = YOLO(r"D:\runway_model\runs\detect\train8\weights\best.pt")

# Run on video (file path OR webcam index 0)
results = model.predict(
    source=r"D:\runway_model\oil spill\12554404_3840_2160_30fps (1).mp4",   # OR 0 for webcam
    show=True,                 # display output
    conf=0.25,                 # confidence threshold
    save=True                  # save output video
)