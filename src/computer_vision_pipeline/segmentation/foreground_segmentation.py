from sklearn.mixture import GaussianMixture
import cv2
import numpy as np


def segment_foreground(fitc_image):
    """
    Predicts a semantic segmentation mask of the cells forground using a threshold selected as
    the mean of two Gaussian Mixture compopnents fitted to the background and foreground pixels intensity respectively.

    Parameters
    ----------
    fitc_image: ndarray
        2D array containing data with float type
        Single channel grayscale morphology (FITC) image
    image_height: int
        image height in pixels
    image_width: int
        image image_width in pixels
    Returns
    -------
    segmentation_result: ndarray
        2D array containing data with boolean type
        predicted cell foreground segmentation mask
    """
    image_height, image_width = fitc_image.shape
    # down sample the image for better performance
    down_sampled_morphology_image = cv2.resize(fitc_image, (image_height // 4, image_width // 4), interpolation=cv2.INTER_AREA)
    # fit a Gaussian mixture model with 2 components
    gmm_classifier = GaussianMixture(n_components=2)
    gmm_classifier.fit(down_sampled_morphology_image.reshape((down_sampled_morphology_image.size, 1)))
    # set the threshold as the average of the connected components' means
    threshold = np.mean(gmm_classifier.means_)
    # A pixel is considered a foreground pixel if its intensity is higher than threshold
    segmentation_result = fitc_image > threshold
    return segmentation_result
