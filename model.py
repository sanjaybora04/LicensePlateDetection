import cv2
import imutils
import numpy as np

model = cv2.dnn.readNetFromCaffe("deploy.prototxt","model.caffemodel")

classNames = {0:"licence",1:"None of the above"}
inputShape = (300,300)
mean=(127.5,127.5,127.5)
scale=0.007843

img = None
blob = None

output = None

def preprocess(image):
    global img,blob
    img = image
    blob = cv2.dnn.blobFromImage(img,scalefactor=scale,size=inputShape,mean=mean,swapRB=True)

def predict():
    global output
    model.setInput(blob)
    output = model.forward()
    if output[0][0][0][1]==0:
        return False
    else:
        return True

# ------------------------------------------ wrap prespective------------------------------------
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    pts = np.float32(pts)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
    return warped
# -----------------------------------------------------------------------------------------------

def process():
    global img
    pt1 = int(output[0][0][0][3]*2000)
    pt2 = int(output[0][0][0][4]*2000)
    pt3 = int(output[0][0][0][5]*2000)
    pt4 = int(output[0][0][0][6]*2000)
    
    img = cv2.resize(img,(2000,2000))
    
    # crop the predicted image with some margin
    img = img[pt2-70:pt4+70,pt1-70:pt3+70]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    gray = cv2.bilateralFilter(gray,11,17,17)
    gray = cv2.Canny(gray,30,20,L2gradient=True)

    keypoints = cv2.findContours(gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,50,True)
        # img = cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        if len(approx) == 4:
            # img = cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            location = approx
            break

    try:
        img = four_point_transform(img,[location[0][0],location[1][0],location[2][0],location[3][0]])
        return {"image":img,"Warning":""}
    except:
        return {"image":img,"Warning":"Edge detection failed, returning cropped image"}
    
def detect(image):
        preprocess(image)
        if predict():
            return process()
        else:
            global img 
            img = cv2.resize(img,(500,500))
            return {"image":img,"Warning":"No plates were detected, returning the original image"}

if __name__ == "__main__":
    pass