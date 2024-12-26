import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_vertical = np.array([[-7, 0, 7],
                            [-20, 0, 20],
                            [-7, 0, 7]])

kernel_horizontal = np.array([[-6, -20, -7],
                              [ 0,  0,  0],
                              [ 6,  20,  7]])

kernel_diagonal = np.array([[ 0,  20,  7],
                            [-6,  0,  6],
                            [-7, -20,  0]])

kernel_minus_45 = np.array([[ 7,  20,  0],
                            [ 6,  0, -6],
                            [ 0, -20, -7]])

vertical_lines = cv2.filter2D(gray, -1, kernel_vertical)

horizontal_lines = cv2.filter2D(gray, -1, kernel_horizontal)

diagonal_lines = cv2.filter2D(gray, -1, kernel_diagonal)

minus_45_lines = cv2.filter2D(gray, -1, kernel_minus_45)

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(vertical_lines, cmap='gray')
plt.title("Vertical Lines")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(horizontal_lines, cmap='gray')
plt.title("Horizontal Lines")
plt.axis("off")

# Plot diagonal lines (top-left to bottom-right)
plt.subplot(3, 2, 4)
plt.imshow(diagonal_lines, cmap='gray')
plt.title("Diagonal Lines (Top-Left to Bottom-Right)")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.imshow(minus_45_lines, cmap='gray')
plt.title("-45 Degree Lines (Top-Right to Bottom-Left)")
plt.axis("off")

# Show the plots
plt.tight_layout()
plt.show()