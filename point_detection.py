import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('forpoint.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pd = np.array([[-1, -1, -1],
               [-1, 8, -1],
               [-1, -1, -1]])
pointdetection = cv2.filter2D(gray, -1, pd)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(pointdetection, cmap='gray')
axes[1].set_title('Point Detection')
axes[1].axis('off')
plt.tight_layout()
plt.show()