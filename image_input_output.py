import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image1.jpg')

fig, axes = plt.subplots(1, 1, figsize=(10, 5))

axes.imshow(image, cmap='gray')
axes.set_title('Output Image')
axes.axis('off')

plt.show()