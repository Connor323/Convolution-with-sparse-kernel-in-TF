import glob
import cv2

files = glob.glob("*.png")

for f in files:
    img = cv2.imread(f)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w/3*2, h/3*2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f, img)
