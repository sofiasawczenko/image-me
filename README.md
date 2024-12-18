# Lane Detection using OpenCV

This project demonstrates the use of **OpenCV** for lane detection on street images. The goal is to process the image, detect edges using the Canny edge detector, apply a mask to focus on the region of interest, and then extract the lane boundaries.

## Requirements

This project requires the following libraries:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Google Colab's file upload functionality (`google.colab`)

You can install the required libraries using `pip` if they are not already installed:

```bash
pip install opencv-python numpy matplotlib
```

## Project Overview

### 1. Import Libraries
We start by importing necessary libraries for image processing and visualization.

```bash
import cv2 as cv
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
```

### 2. Upload Image
You can upload your own image for lane detection using the file upload widget in Google Colab.

```bash
upload = files.upload()
### 3. Image Preparation
We load the image, check its type, and print its dimensions.
```

```bash
imagem = (r'lane_detection.jpeg')
img_cv = cv.imread(imagem)
print(type(img_cv))
print('Tamanho da imagem: ', img_cv.shape)
```

###4. Convert Image to RGB
Next, we convert the image from BGR (default in OpenCV) to RGB for visualization.

```bash
img = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
plt.imshow(img)
```

### 5. Convert Image to Grayscale
Since edge detection works best with grayscale images, we convert the image to grayscale.

```bash
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap=plt.cm.gray)
```

### 6. Canny Edge Detection
We use the Canny edge detection algorithm to detect edges in the grayscale image.

```bash
linhas = cv.Canny(img_gray, 100, 200)
plt.imshow(linhas, cmap='gray')
plt.show()
```

### 7. Automatic Threshold for Canny Edge Detection
A function auto_canny is created to automatically determine the threshold values for Canny edge detection.

```bash
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    return edged

img_canny = auto_canny(img)
plt.imshow(img_canny)
8. Creating a Mask for Region of Interest
We define a polygon that isolates the region of interest (the area where the lane is located) and apply it as a mask.
```

```bash
mask = np.zeros_like(img_canny)
h, w = img_canny.shape
pts = np.array([[50,h],[300,240],[400,240],[650,h]], dtype=np.int32)
mask_filled = cv.fillPoly(mask, [pts], (255, 255, 255))
plt.imshow(mask_filled)
```

### 9. Apply Mask to Edge Image
We apply the mask to the edge-detected image to focus on the region of interest.

```bash
masked_image = cv.bitwise_and(img_canny, mask_filled)
plt.imshow(masked_image)
```

Image Transformation Techniques
In addition to lane detection, other image processing techniques like translation, rotation, and flipping are demonstrated.

Translation
We perform a translation on the image by shifting it in both the X and Y directions.

```bash
deslocamento = np.float32([[1,0,100],[0,1,100]])
img_trans = cv.warpAffine(img, deslocamento, (largura, altura))
plt.imshow(img_trans)
Rotation
We rotate the image around its center by a specified angle.
```

```bash
rotacao = cv.getRotationMatrix2D(ponto, 30, 1.0)
img_rot = cv.warpAffine(img, rotacao, (largura, altura))
plt.imshow(img_rot)
Flipping
We flip the image along different axes (X, Y, or both).
```

```bash
espelhar = cv.flip(img, -1)
plt.imshow(espelhar)
```

## Conclusion
This project showcases basic image processing techniques using OpenCV, including edge detection, region of interest masking, and image transformations. These techniques are essential for many computer vision tasks such as lane detection in autonomous driving systems.

## License
This project is open-source and available under the MIT License.

