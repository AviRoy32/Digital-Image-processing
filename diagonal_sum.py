import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('image1.jpg')

if image.ndim == 3:
    image = np.mean(image, axis=2)

main_diagonal = np.diag(image)

anti_diagonal = np.diag(np.fliplr(image))

diagonal_sum = np.sum(main_diagonal) + np.sum(anti_diagonal)
if image.shape[0] == image.shape[1]:
    diagonal_sum -= image[image.shape[0] // 2, image.shape[1] // 2]

print(f"Sum of diagonal elements: {diagonal_sum}")