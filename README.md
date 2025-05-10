# AI-Powered Object Detection with Faster R-CNN

This project demonstrates how to use a pre-trained Faster R-CNN model to perform object detection on images. The model is capable of identifying various objects within an image and drawing bounding boxes around them. The detected objects are labeled with their respective classes, and the output is saved as a new image.

## Requirements

To run the project, you'll need to install the following dependencies:

1. **opencv-python-headless**: For reading and processing images.
2. **torch**: PyTorch framework.
3. **torchvision**: Provides the pre-trained Faster R-CNN model.
4. **matplotlib**: For displaying the image with detected objects.

You can install all required dependencies by running the following command:

```bash
pip install -r requirements.txt
Alternatively, you can install the dependencies individually using pip:

bash
Copy
Edit
pip install opencv-python-headless
pip install torch torchvision
pip install matplotlib
