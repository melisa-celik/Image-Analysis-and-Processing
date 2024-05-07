import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageDraw
from PythonImplementations.NumberRecognition_MNIST.LeNet import *
from PythonImplementations.HOG.HOG import *

def extractHogFeatures(image, hog):
    """
    Extract HOG features for the entire image using a sliding window approach.
    """
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHeight, imageWidth = grayImage.shape

    patchSize = (28, 28)
    stride = 4  # Stride for sliding window

    hogFeatures = []
    positions = []

    for y in range(0, imageHeight - patchSize[0] + 1, stride):
        for x in range(0, imageWidth - patchSize[1] + 1, stride):
            patch = grayImage[y:y+patchSize[0], x:x+patchSize[1]]
            hog_feature = hog.computeHogFeatures(patch)
            hogFeatures.append(hog_feature)
            positions.append((x, y))

    return hogFeatures, positions

def detectNumbers(image, model, hog):
    """
    Detect numbers in the image using a sliding window with HOG features and a neural network model.
    """
    hogFeatures, positions = extractHogFeatures(image, hog)

    predictions = []

    for hogFeature in hogFeatures:
        hogTensor = torch.FloatTensor(hogFeature).unsqueeze(0)  # Convert to tensor
        output = model(hogTensor)
        _, predicted = torch.max(output, 1)
        predictions.append(predicted.item())

    for idx, pred in enumerate(predictions):
        if pred != 0:
            x, y = positions[idx]
            draw = ImageDraw.Draw(image)
            draw.rectangle([x, y, x + 28, y + 28], outline="red")

    return image

def main():
    model = LeNet()
    model.load_state_dict(torch.load('./model.path'))

    hog = HOG()

    image = cv2.imread('PythonImplementations/NumberRecognition_MNIST/numbers.png')

    resultImage = detectNumbers(image, model, hog)

    cv2.imshow('Detected Numbers', resultImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()