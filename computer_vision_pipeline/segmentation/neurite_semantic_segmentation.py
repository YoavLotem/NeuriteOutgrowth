from computer_vision_pipeline.common import IMAGE_SHAPE
import cv2
import numpy as np


def apply_post_processing(binary_prd, thr=100):
    """
    Performs post-processing on a binary segmentation prediction by
    removing connected components whose size [pixels] is smaller than a provided threshold

    Parameters
    ----------
    binary_prd: ndarray
        2D array containing data with boolean type
        Binary segmentation prediction before pre-processing
    thr: int or float, default=100
        Indicates the minimum size of a connected component that will remain

    Returns
    -------
    new_binary_prd: ndarray
        2D array containing data with boolean type
        Binary segmentation prediction after removing connected components smaller than threshold
    """

    # extract the connected components in the image and their respective statistics
    # the first component is the background so we remove it
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_prd.astype('uint8'), connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # initializing a new binary prediction array and inserting it with the connected components larger than threshold
    new_binary_prd = np.zeros(IMAGE_SHAPE)
    for i in range(0, nb_components):
        if sizes[i] >= thr:
            new_binary_prd[output == i + 1] = 1
    return new_binary_prd


def apply_pre_processing(morphology_image):
    """
    Applies pre-processing to the input morphology image(FITC) prior to inference using the neurite segmentation model

    Parameters
    ----------
    morphology_image: ndarray
        2D array containing data with float type
        Single channel grayscale morphology image

    Returns
    -------
    X: ndarray
        4D array containing data with float type
        Input image after pre-processing
    """
    # applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve contrast
    # than the same pre-processing procedure as in training: subtracting the mean and dividing by 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    morphology_image = clahe.apply(morphology_image)
    morphology_image = morphology_image - np.mean(morphology_image)
    morphology_image = morphology_image / 255
    # adding more channels to turn image into a "batch" for deep learning model
    X = morphology_image[np.newaxis, :, :, np.newaxis]
    return X


def segment_neurites(morphology_image, neurite_model):
    """
    Predicts a semantic segmentation mask of neurite's pixels in the
    input morphology image using the neurite segmentation model.

    Parameters
    ----------
    morphology_image: ndarray
        2D array containing data with float type
        Single channel grayscale morphology image
    neurite_model: Keras model
        CNN for neurite semantic segmentation

    Returns
    -------
    segmentation_result: ndarray
        2D array containing data with boolean type
        predicted neurite segmentation mask

    """

    X = apply_pre_processing(morphology_image)

    # using the neurite segmentation model for predicting the neurites pixels (probability for each pixel),
    # thresholding the probability map at 0.5 and applying post-processing
    prd = neurite_model.predict(X, steps=1)
    binary_prd = prd[0, :, :, 0] > 0.5
    segmentation_result = apply_post_processing(binary_prd)
    return segmentation_result
