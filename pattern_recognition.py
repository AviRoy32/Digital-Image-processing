import cv2
import matplotlib.pyplot as plt

# Load images
main_image = cv2.imread("pust.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("pust lake.jpg", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if main_image is None or template is None:
    print("Error: One or both images could not be loaded. Check the file paths.")
    exit()

# Validate template size
if template.shape[0] > main_image.shape[0] or template.shape[1] > main_image.shape[1]:
    print("Error: Template size is larger than the main image.")
    exit()

# Template dimensions
w, h = template.shape[::-1]

# Perform template matching
result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)

# Get min/max values and locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle around the matched region
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
matched_image = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)

# Display results
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Main Image")
plt.imshow(main_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Template")
plt.imshow(template, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Matched Result")
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
