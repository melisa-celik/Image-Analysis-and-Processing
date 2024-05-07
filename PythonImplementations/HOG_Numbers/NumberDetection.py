from PythonImplementations.HOG.HOG import HOG
import torch
import cv2
from PIL import Image, ImageDraw
from PythonImplementations.NumberRecognition_MNIST.LeNet import LeNet, preprocessImage
import numpy as np

class NumberDetector:
    def __init__(self, model_path):
        self.hog = HOG()
        self.model = LeNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


    def slidingWindows(self, image, patchSize, stride):
        if len(image.shape) == 2:
            height, width = image.shape
        elif len(image.shape) == 3:
            height, width, _ = image.shape
        else:
            raise ValueError("Unsupported image format")

        patchHeight, patchWidth = patchSize

        for y in range(0, height - patchHeight + 1, stride):
            for x in range(0, width - patchWidth + 1, stride):
                patch = image[y:y + patchHeight, x:x + patchWidth]

                patch = np.uint8(patch)
                if len(patch.shape) == 3:
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                # Resize patch to (28x28)
                pil_patch = Image.fromarray(patch)
                pil_patch = pil_patch.resize((28, 28))
                resized_patch = np.array(pil_patch)

                yield (x, y, resized_patch)

    def detectNumbers(self, image):
        print("Image Shape:", image.shape)
        print("Image Data Type:", image.dtype)

        if len(image.shape) > 2 and image.shape[2] > 1:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayImage = image

        detectedNumbers = []
        patchSize = (28, 28)
        stride = 4

        for x, y, patch in self.slidingWindows(grayImage, patchSize, stride):
            hogFeatures = self.hog.computeHogFeatures(patch)
            tensor = torch.tensor(hogFeatures, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tensor)

            _, predicted = torch.max(output, 1)

            if predicted.item() != 0:
                detectedNumbers.append((x, y, predicted.item()))

        return detectedNumbers

    def visualizeDetection(self, imagePath, detections):
        image = Image.open(imagePath)
        draw = ImageDraw.Draw(image)

        for x, y, number in detections:
            draw.rectangle([x, y, x + 28, y + 28], outline="red")

        image.show()


def main():
    imagePath = 'numbers.png'
    modelPath = './model.pth'

    detector = NumberDetector(modelPath)
    inputTensor = preprocessImage(imagePath)
    inputImage = inputTensor.squeeze(0).permute(1, 2, 0).numpy()
    detections = detector.detectNumbers(inputImage)

    if detections:
        print("Detected numbers:")
        for x, y, number in detections:
            print(f"Number {number} detected at position ({x}, {y})")

        detector.visualizeDetection(imagePath, detections)
    else:
        print("No numbers detected in the image.")

if __name__ == '__main__':
    main()