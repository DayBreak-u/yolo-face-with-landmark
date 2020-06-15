import cv2
img = cv2.imread("../test_imgs/inputs/2.jpg")



m = cv2.getRotationMatrix2D(angle=10, center=(img.shape[1] / 2, img.shape[0] / 2), scale=2.0)
img = cv2.warpAffine(img, m[:2], dsize=(img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

cv2.imwrite('test.jpg',img)