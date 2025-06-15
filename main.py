import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tensorflow as tf

def classify_and_segment_image(cnn_model_path, yolo_model_path, image_path, input_size=(150, 150), conf=0.5, output_dir='output'):
    """
    
    """
    try:
        # Validate inputs
        if not os.path.exists(cnn_model_path):
            raise FileNotFoundError(f"CNN model not found: {cnn_model_path}")
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load models
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        yolo_model = YOLO(yolo_model_path)
        
        # Print YOLO model class names for verification
        print(f"YOLO model class names: {yolo_model.names}")

        # Load and preprocess image for CNN
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # CNN preprocessing
        cnn_image = cv2.resize(image, input_size)
        cnn_image = cv2.cvtColor(cnn_image, cv2.COLOR_BGR2RGB)
        cnn_image = cnn_image / 255.0
        cnn_image = np.expand_dims(cnn_image, axis=0)

        # Classify with CNN
        class_names = ['glioma', 'meningioma', 'pituitary', 'no tumor']
        predictions = cnn_model.predict(cnn_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        classification_result = {
            'class': class_names[predicted_class_idx],
            'confidence': confidence
        }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Segment with YOLO
        yolo_results = yolo_model.predict(source=image, imgsz=640, conf=conf)
        
        # Debug YOLO detections
        if yolo_results and yolo_results[0].boxes:
            boxes = yolo_results[0].boxes.data.cpu().numpy()
            detected_classes = [int(box[5]) for box in boxes]
            detected_class_names = [yolo_model.names[cls] for cls in detected_classes]
            print(f"Total YOLO detections: {len(boxes)}")
            print(f"Detected classes: {detected_classes}")
            print(f"Detected class names: {detected_class_names}")
        else:
            print("No detections by YOLO")

        # Filter YOLO detections based on CNN prediction
        filtered_boxes = []
        if classification_result['class'] != 'no tumor':
            # Map CNN class names to YOLO class indices
            class_mapping = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
            target_class_id = class_mapping.get(classification_result['class'], -1)
            print(f"Target class ID for {classification_result['class']}: {target_class_id}")
            
            if target_class_id != -1 and yolo_results and yolo_results[0].boxes:
                boxes = yolo_results[0].boxes.data.cpu().numpy()
                for box in boxes:
                    class_id = int(box[5])
                    if class_id == target_class_id:
                        filtered_boxes.append(box)
        
        tumor_detected = bool(filtered_boxes)

        # Generate annotated image with filtered bounding boxes
        annotated_image = image.copy()
        for box in filtered_boxes:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Convert to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Save annotated image if tumors detected
        output_image_path = None
        if tumor_detected:
            output_image_name = f"annotated_{os.path.basename(image_path)}"
            output_image_path = os.path.join(output_dir, output_image_name)
            cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR))
            print(f"Annotated image saved to: {output_image_path}")

        # Display image
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image_rgb)
        plt.title(
            f"CNN: {classification_result['class']} ({classification_result['confidence']:.2%})\n"
            f"YOLO: {'Tumor detected' if tumor_detected else 'No tumor detected for the predicted class'}",
            fontsize=14
        )
        plt.axis('off')
        plt.show()

        return classification_result, output_image_path

    except Exception as e:
        print(f"Error during classification and segmentation: {e}")
        return None, None

def main():
    # Configurable paths
    cnn_model_path = 'braintumor.h5'
    yolo_model_path = 'best.pt'
    image_path = 'images/test/1046_jpg.rf.26dd526422308d6d9e989d3308a9ee03.jpg'
    output_dir = 'output'
    input_size = (150, 150)

    # Run classification and segmentation
    print("Processing image...")
    classification_result, output_image_path = classify_and_segment_image(
        cnn_model_path, yolo_model_path, image_path, input_size=input_size, conf=0.5, output_dir=output_dir
    )

    if classification_result:
        print(f"Classification Result: {classification_result['class']} (Confidence: {classification_result['confidence']:.2%})")
        if output_image_path:
            print(f"Output image: {output_image_path}")
        else:
            print("No tumors detected by YOLO for the predicted class.")
    else:
        print("Failed to process image.")

main()