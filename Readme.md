# Licence Plate Segmentation

* This module uses a prototxt file and a caffemodel file to detect the license plate
* It uses edge detection to get the exact coordinates of the license plate
* Then it uses wrap prespective method of opencv to change the prespective of image

## How to use :-

* Import model.py in your code

```shell
import model.py
```

* The detect function takes path of image as input

```shell
result = model.detect("image.jpg")
```

### Result Format
* It returns a dictionary with 2 variables- the result image(_numpy array_), and a warning(_warnings are described below_)

```shell
{
"image": <__result_image__>,
"Warning": <__warning__>,
}
```

### Warnings:
* Warnings are returned in form of a string
* If there are no warnings, an empty string is returned
* If there is a warning, the message is returned in the "Warning" variable.

### Sample Code:

```shell
import model
import cv2

image = "***path_to_image***"

img = cv2.imread(image)

cv2.imshow('frame1',cv2.resize(img,(500,500)))
result = model.detect(image)
cv2.imshow('frame',result["image"])
if result["Warning"]:
    print(result["Warning"])
cv2.waitKey(0)
```
