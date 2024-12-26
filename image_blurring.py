import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image1.jpg')

gaussian_blur = cv2.GaussianBlur(image, (7, 7), 0)

# Convert to RGB for matplotlib
#gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(gaussian_blur, cmap='gray')
axes[1].set_title('Blurred Image')
axes[1].axis('off')

# Show the images
plt.show()