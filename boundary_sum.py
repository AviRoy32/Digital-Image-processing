import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('image1.jpg')

if image.ndim == 3:
    image = np.mean(image, axis=2)

top_row = image[0, :]
bottom_row = image[-1, :]
left_column = image[:, 0]
right_column = image[:, -1]

boundary_sum = np.sum(top_row) + np.sum(bottom_row) + np.sum(left_column) + np.sum(right_column)

boundary_sum -= (image[0, 0] + image[0, -1] + image[-1, 0] + image[-1, -1])

print(f"Sum of boundary elements: {boundary_sum}")