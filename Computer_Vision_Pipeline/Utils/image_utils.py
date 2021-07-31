import cv2
import imgaug.augmentables.segmaps as bla
from skimage import measure
import numpy as np
import os

def Backscatter_flag(dapi):
    #  convert the image to grayscale, and blur it
    gray = cv2.cvtColor(dapi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # threshold the image to reveal light regions in the blurred image
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, connectivity=2, background=0)
    flag = 0
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 10000:
            flag += numPixels
            break
    return flag


def overlay(image, mask, name, path):
    mask2 = np.zeros((2048, 2048, 3), dtype=bool)
    mask2[:, :, 0] = mask.astype('bool')
    image2 = np.zeros((2048, 2048, 3))
    image2[:, :, 0], image2[:, :, 1], image2[:, :, 2] = image, image, image
    segmap = bla.SegmentationMapOnImage(mask2, shape=image2.shape)
    b = segmap.draw_on_image(image2.astype('uint8'), alpha=0.3)
    n = os.path.join(path, name + 'overlay.png')
    cv2.imwrite(n, b)