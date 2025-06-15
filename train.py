from ultralytics import YOLO

# 1. Load a pretrained YOLO11-nano detection model
model = YOLO("yolo11n.pt")

# 2. Train on COCO‑8 for 100 epochs
#    If you want the built‑in example, use coco8.yaml; otherwise, point to your own data.yaml
train_results = model.train(
    data="train.yaml",   # or "data.yaml" for your own dataset
    epochs=50,patience=20, batch= -1, optimizer='auto'            # You can specify batch size if you want (default is 16)
    device="cpu",        # or "0" / [0,1] for GPU(s)
    name="brain_tumor_yolo11n"
)

# 3. Save the best weights to a fixed filename
model.save("Brain_tumor_yolo11n.pt")

# 4. Evaluate on the validation set
metrics = model.val()  # prints and returns precision, recall, mAP, etc.

# 5. Inference on a single image
results = model("D:/Phyton Work/Brain Tumor Segmentation/images/test/10_jpg.rf.efaf1af26de11dabdda3214f4457c931.jpg", conf=0.25)  # set confidence threshold as needed
results[0].show()  # display with bounding boxes

# 6. Export to ONNX for deployment
onnx_path = model.export(format="onnx")  # returns path like "runs/detect/train/weights/last.onnx"
print(f"ONNX model saved to: {onnx_path}")
