import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('image.jpg')
top, bottom, left, right = 100, 100, 100, 100

padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
axes[0].set_title('Original Image')
axes[0].axis('off')

# Padded image
axes[1].imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
axes[1].set_title('Padded Image')
axes[1].axis('off')

# Show the images
plt.show()