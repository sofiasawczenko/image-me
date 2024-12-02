# Computer Vision with OpenCV

This repository contains Jupyter notebooks demonstrating basic and advanced image processing and computer vision techniques using OpenCV in Python. The notebooks cover a variety of tasks, from basic image manipulation to more complex problems like lane detection.

## Notebooks

### 1. **processamento_de_imagem_OpenCV**
This notebook introduces the basic concepts and techniques in image processing using OpenCV. It covers operations like reading, displaying, and manipulating images, as well as applying filters and transformations. It's a great starting point for anyone looking to learn OpenCV.

### 2. **ComputerVision1.ipynb: utilizando o OpenCV para processamento de imagem**
This notebook demonstrates how to use OpenCV for general image processing tasks. The key concepts explored in this notebook include:
- Converting images to grayscale.
- Applying Gaussian blur for noise reduction.
- Performing edge detection using the Canny algorithm.

#### Key Code Example:
```python
import cv2
import numpy as np

# Reading an image
image = cv2.imread('image.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Displaying the image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applying Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred_image, 100, 200)

# Display the result
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
