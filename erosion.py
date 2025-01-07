import cv2
import matplotlib.pyplot as plt

# Load a binary image
image = cv2.imread('pust.jpg', cv2.IMREAD_GRAYSCALE)

# Define the structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Rectangular 3x3 kernel

# Apply erosion
eroded_image = cv2.erode(image, kernel, iterations=1)

# Display the original and eroded images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Eroded Image")
plt.imshow(eroded_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()