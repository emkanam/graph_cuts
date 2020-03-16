from graphCut.graphModel import GraphModel
from graphCut.alphaExpansion import alpha_expansion
import cv2

# read image
filename = '../data/img_1.png'
image = cv2.imread(filename, 0)
cv2.imshow('original image', image)
cv2.waitKey(0)
print(image.shape)

# segment using alpha expansion
# labels = alpha_expansion(image)

# display segmented result
# cv2.imshow('segmented image', labels)
# cv2.waitKey(0)