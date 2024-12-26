import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("image.jpg", cv.IMREAD_GRAYSCALE)

row, col = img.shape

y = np.zeros((256), np.uint64)

for i in range(row):
    for j in range(col):
        y[img[i, j]] += 1

x = np.arange(0, 256)

x_modified = np.arange(0, 256, 5)

y_modified = [y[i] for i in x_modified]

plt.subplot(1, 2, 1)
plt.bar(x, y, color="red", align="center")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Grayscale Histogram')

plt.subplot(1, 2, 2)
plt.bar(x_modified, y_modified, color="red", align="center")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Modified Grayscale Histogram')

plt.show()