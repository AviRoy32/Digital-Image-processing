import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('image.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create the negative image
negative_image = 255 - image_rgb
cv2.imwrite('output_negative_image.jpg', cv2.cvtColor(negative_image, cv2.COLOR_RGB2BGR))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Negative image
axes[1].imshow(negative_image)
axes[1].set_title('Negative Image')
axes[1].axis('off')

plt.show()