import cv2
from skimage import measure
import numpy as np


def quntifyBackscatter(dapi_im):
    """
    Detects whether backscatter exists in the image.
    Parameters
    ----------
    dapi_im: ndarray
        2D array containing data with float type
        Single channel grayscale Nuclei(DAPI) image

    Returns
    -------
    count: int
        The number of pixels regarded as backscatter.
    """
    #  convert the image to grayscale, and blur it
    gray = cv2.cvtColor(dapi_im, cv2.COLOR_BGR2GRAY)
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
    count = 0
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
        # large, then add it to our count of "large blobs"
        if numPixels > 10000:
            count += numPixels
    return count



