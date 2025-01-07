import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a binary image
image = cv2.imread('pust.jpg', cv2.IMREAD_GRAYSCALE)

# Define the structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Rectangular 3x3 kernel

# Apply dilation
dilated_image = cv2.dilate(image, kernel, iterations=1)

# Display the original and dilated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Dilated Image")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()