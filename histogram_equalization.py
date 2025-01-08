import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("pust.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Please ensure 'cat2.jpg' is in the working directory.")

levels = 256
histogram = [0] * levels
rows, cols = image.shape

for i in range(rows):
    for j in range(cols):
        pixel_value = image[i, j]
        histogram[pixel_value] += 1

cdf = [0] * levels
cdf[0] = histogram[0]

for i in range(1, levels):
    cdf[i] = cdf[i - 1] + histogram[i]

total_pixels = rows * cols
normalized_cdf = [int(c * (levels - 1) / total_pixels) for c in cdf]

equalized_image = np.zeros_like(image)

for i in range(rows):
    for j in range(cols):
        original_value = image[i, j]
        equalized_image[i, j] = normalized_cdf[original_value]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()