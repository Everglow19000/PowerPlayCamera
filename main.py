# %%
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import pytesseract


# %%
img = cv2.imread('./src/everglow.jpeg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

THRESHOLD1 = 500
THRESHOLD2 = 500
canny = cv2.Canny(img, THRESHOLD1, THRESHOLD2)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# contours, hierarchy = cv2.findContours(
#     canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# %%
msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
dlt = cv2.dilate(msk, krn, iterations=1)
thr = 255 - cv2.bitwise_and(dlt, msk)

d = pytesseract.image_to_string(thr, config="--psm 10")
print("Result: " + d)

cv2.imshow("Image", thr)
cv2.waitKey(0)
cv2.destroyAllWindows()
