import numpy as np
import cv2

class HOG:
    def __init__(self, cellSize=8, numBins=9):
        self.cellSize = cellSize
        self.numBins = numBins

    def computeGradients(self, image):
        fx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        fy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        magnitude = np.linalg.norm([fx, fy], axis=0)
        orientation = np.arctan2(fy, fx) * (180 / np.pi) % 180

        return magnitude, orientation

    def createHistograms(self, magnitude, orientation):
        height, width = magnitude.shape
        numCells_y = height // self.cellSize
        numCells_x = width // self.cellSize
        histograms = []

        bin_edges = np.linspace(0, 180, self.numBins + 1)  # Define bin edges from 0 to 180 degrees
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # Compute bin centers

        for y in range(numCells_y):
            for x in range(numCells_x):
                cellMagnitude = magnitude[y * self.cellSize:(y + 1) * self.cellSize,
                                           x * self.cellSize:(x + 1) * self.cellSize]
                cellOrientation = orientation[y * self.cellSize:(y + 1) * self.cellSize,
                                              x * self.cellSize:(x + 1) * self.cellSize]

                hist, _ = np.histogram(cellOrientation, bins=bin_edges, weights=cellMagnitude)

                histograms.append(hist)

        return histograms

    def normalizeBlocks(self, histograms, blockSize=2):
        numCells_x = numCells_y = int(np.sqrt(len(histograms)))
        normalizedFeatures = []

        for y in range(numCells_y - blockSize + 1):
            for x in range(numCells_x - blockSize + 1):
                blockHist = []
                for i in range(blockSize):
                    for j in range(blockSize):
                        blockHist.extend(histograms[(y + i) * numCells_x + (x + j)])

                blockHist /= np.linalg.norm(blockHist) + 1e-5
                normalizedFeatures.append(blockHist)

        return normalizedFeatures

    def computeHogFeatures(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        magnitude, orientation = self.computeGradients(image)

        histograms = self.createHistograms(magnitude, orientation)

        blockSize = 2  # Adjust as needed
        hogFeatures = self.normalizeBlocks(histograms, blockSize)

        return hogFeatures


def main():
    image = cv2.imread("hog_test.png", cv2.IMREAD_COLOR)
    hog = HOG()
    features = hog.computeHogFeatures(image)

    print("HOG Feature vector:", features)

    print("HOG Feature vector size:", len(features[0]))

if __name__ == "__main__":
    main()
