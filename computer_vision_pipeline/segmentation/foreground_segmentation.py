from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
from computer_vision_pipeline.common import IMAGE_WIDTH, IMAGE_HEIGHT


def segment_foreground(morphology_image):
    """
    Predicts a semantic segmentation mask of the cells forground using a threshold selected as
    the mean of two Gaussian Mixture compopnents fitted to the background and foreground pixels intensity respectively.

    Parameters
    ----------
    morphology_image: ndarray
        2D array containing data with float type
        Single channel grayscale morphology image

    Returns
    -------
    segmentation_result: ndarray
        2D array containing data with boolean type
        predicted cell foreground segmentation mask
    """

    # down sample the image for better performance
    down_sampled_morphology_image = cv2.resize(morphology_image, (IMAGE_HEIGHT/4, IMAGE_WIDTH/4), interpolation=cv2.INTER_AREA)
    # fit a Gaussian mixture model with 2 components
    gmm_classifier = GaussianMixture(n_components=2)
    gmm_classifier.fit(down_sampled_morphology_image.reshape((down_sampled_morphology_image.size, 1)))
    # set the threshold as the average of the connected components' means
    threshold = np.mean(gmm_classifier.means_)
    # A pixel is considered a foreground pixel if its intensity is higher than threshold
    segmentation_result = morphology_image > threshold
    return segmentation_result
