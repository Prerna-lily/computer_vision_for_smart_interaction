import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt

# Requirements to install using pip:
# pip install opencv-python-headless
# pip install torch torchvision
# pip install matplotlib

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define a list of COCO classes
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Apply transformations
transform = T.Compose([
    T.ToTensor()
])

def predict(image_path):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image file '{image_path}'.")
        return

    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(image)

    # Extract labels and bounding boxes
    for idx, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
        if score >= 0.5:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.tolist())
            label_name = COCO_CLASSES[label]
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_image, f"{label_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, orig_image)
    print(f"Output saved as {output_path}")

    # Display using matplotlib
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Objects')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = "items.jpeg"
    predict(image_path)
