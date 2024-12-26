import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('image.jpg')

# Apply the log transform
c = 255 / np.log(1 + np.min(image))  # Scaling constant
log_transformed = c * (np.log(1 + image))
log_transformed = np.array(log_transformed, dtype=np.uint8)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Log-transformed image
axes[1].imshow(log_transformed, cmap='gray')
axes[1].set_title('Log Transformed Image')
axes[1].axis('off')

# Show the images
plt.show()