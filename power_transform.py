import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')
image_normalized = image / 255.0

gamma = 5.0
c = 1
gamma_transformed = c * np.power(image_normalized, gamma)

gamma_transformed = np.uint8(gamma_transformed * 255)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Gamma-transformed image
axes[1].imshow(gamma_transformed, cmap='gray')
axes[1].set_title(f'Gamma Transformed (γ={gamma})')
axes[1].axis('off')

# Show the images
plt.show()