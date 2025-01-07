import cv2
import matplotlib.pyplot as plt

image = cv2.imread('pust.jpg', cv2.IMREAD_GRAYSCALE)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Opened Image")
plt.imshow(opened_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()