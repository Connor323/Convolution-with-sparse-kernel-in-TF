import glob
import cv2

files = glob.glob("*.png")

for f in files:
    img = cv2.imread(f)
    img = cv2.resize(img, (100, 200), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f, img)
