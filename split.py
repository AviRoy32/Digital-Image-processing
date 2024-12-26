import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image using OpenCV
image = cv2.imread('image.jpg')

# Convert the image from BGR to RGB for proper display with matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the midpoints of the width and height
mid_x, mid_y = width // 2, height // 2

# Split the image into four parts
top_left = image[:mid_y, :mid_x]
top_right = image[:mid_y, mid_x:]
bottom_left = image[mid_y:, :mid_x]
bottom_right = image[mid_y:, mid_x:]

# Merge the top halves (top-left and top-right)
top_half = np.concatenate((top_left, top_right), axis=1)

# Merge the bottom halves (bottom-left and bottom-right)
bottom_half = np.concatenate((bottom_left, bottom_right), axis=1)

# Merge the top and bottom halves to get the complete image
merged_image = np.concatenate((top_half, bottom_half), axis=0)

# Use Matplotlib to display the four parts
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Top-left
axs[0, 0].imshow(top_left)
axs[0, 0].set_title('Top Left')
axs[0, 0].axis('off')

# Top-right
axs[0, 1].imshow(top_right)
axs[0, 1].set_title('Top Right')
axs[0, 1].axis('off')

# Bottom-left
axs[1, 0].imshow(bottom_left)
axs[1, 0].set_title('Bottom Left')
axs[1, 0].axis('off')

# Bottom-right
axs[1, 1].imshow(bottom_right)
axs[1, 1].set_title('Bottom Right')
axs[1, 1].axis('off')

# # Leave one subplot blank for symmetry
axs[2, 1].axis('off')

# Display the merged image
axs[2, 0].imshow(merged_image)
axs[2, 0].set_title('Merged Image')
axs[2, 0].axis('off')

# Show the plots
plt.tight_layout()
plt.show()
