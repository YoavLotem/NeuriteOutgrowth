from Computer_Vision_Pipeline.common import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_SHAPE
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np

def applyPostProcessing(binary_prd, thr=100):
    """
    Performs post-processing on a binary segmentation prediction by
    removing connected components whose size [pixels] is smaller than a provided threshold

    Parameters
    ----------
    binary_prd: ndarray
        2D array containing data with boolean type
        Binary segmentation prediction before pre-processing
    thr: int or float
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

def applyPreProcesing(morphology_image):
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

def segmentNeurites(morphology_image, neurite_model):
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

    X = applyPreProcesing(morphology_image)

    # using the neurite segmentation model for predicting the neurites pixels (probability for each pixel),
    # thresholding the probability map at 0.5 and applying post-processing
    prd = neurite_model.predict(X, steps=1)
    binary_prd = prd[0, :, :, 0] > 0.5
    segmentation_result = applyPostProcessing(binary_prd)
    return segmentation_result

def segmentForeground(morphology_image):
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


def segmentNucleiByQuarters(dapi_im, nucModel):
    """
    Performs Nuclei instance segmentation to the DAPI image by dividing it to 4 quarters
    and performing instance segmentation to each quarter seperately, then combining the results.
    This procedure allows the model to detect a larger amount of cells in the image as the
    limitation of the model is up to 1000 cells, this procedure allows the model to detect up to 4000 cells

    Parameters
    ----------
    dapi_im: ndarray
        2D array containing data with float type
        Single channel grayscale Nuclei(DAPI) image

    nucModel: Instance of class MaskRCNN
        Mask RCNN model for Nuclei instance segmentation

    Returns
    -------
    nuclei_segmentation: ndarray
        2D array containing data with int type
        Nuclei instance segmentation results for the entire DAPI image, each integer represents a different nucleus
    nuclei_count: int
        Number of overall nuclei detected in the DAPI image
    """

    # Initialize an image to aggregate the instance segmentation from all 4 quarters and a global
    # nuclei_count to keep track of individual nuclei numbers
    nuclei_segmentation = np.zeros(IMAGE_SHAPE)
    nuclei_count = 1
    for i in range(2):
        for j in range(2):
            # detect the nuclei in a quarter of the image
            quarter_dapi = dapi_im[i * (IMAGE_HEIGHT/2): (i + 1) * (IMAGE_HEIGHT/2), j * (IMAGE_WIDTH/2): (j + 1) * (IMAGE_WIDTH/2), :]
            results = nucModel.detect([quarter_dapi], verbose=0)
            mask_per_nucleus = results[0]['masks']  # a single channel for each detected nuclei with booleans indicating its location
            num_nuclei = np.shape(mask_per_nucleus)[2]
            # aggregate the individual detected instance segmentation masks into a single 2d image
            nuclei_segmentation_quarter = np.zeros((IMAGE_HEIGHT/2, IMAGE_WIDTH/2))
            for idx in range(num_nuclei):
                mask = mask_per_nucleus[:, :, idx]
                nuclei_segmentation_quarter[mask] = nuclei_count
                nuclei_count += 1
            # insert the instance segmentation result of a quarter image in the total image results
            nuclei_segmentation[i * (IMAGE_HEIGHT/2): (i + 1) * (IMAGE_HEIGHT/2), j * (IMAGE_WIDTH/2): (j + 1) * (IMAGE_WIDTH/2)] = nuclei_segmentation_quarter
    return nuclei_segmentation, nuclei_count

def getSplittedNucleiIndices(mask_per_nucleus_in_border):
    """
    Detects which nucleus in the DAPI image is overlapping with the quarters borderline

    Parameters
    ----------
    mask_per_nucleus_in_border: ndarray
        Array with a 2D binary mask for every individual nucleus detected close to the borderline of the quarters

    Returns
    -------
    splitted_nuclei_indices: ndarray
        1D array with boolean type
        indicates which of the binary masks in mask_per_nucleus_in_border belongs to a nucleus that is splitted in the
        segmentation results of segmentNucleiByQuarters (might be splitted into multiple nuclei)
    """
    # initialize a binary 2D array with the same size as the DAPI image with ones on the borderline
    borderline = np.zeros(IMAGE_SHAPE)
    borderline[(IMAGE_HEIGHT/2)-1: (IMAGE_HEIGHT/2)+1, :] = 1
    borderline[:, (IMAGE_WIDTH/2)-1: (IMAGE_WIDTH/2)+1] = 1
    splitted_nuclei_indices = []
    # iterate other every mask of a detected nucleus in the proximity of the borderline and check if it overlaps
    # with the borderline
    for i in range(np.shape(mask_per_nucleus_in_border)[2]):
        if np.any(np.logical_and(borderline, mask_per_nucleus_in_border[:, :, i])):
          splitted_nuclei_indices.append(True)
        else:
          splitted_nuclei_indices.append(False)
    splitted_nuclei_indices = np.array(splitted_nuclei_indices)
    return splitted_nuclei_indices

def findSplittedNuclei(dapi_im, nucModel):
    """
    Segments nuclei in the image that are close to the quarters borderlines (due to the image being divided to four parts
    in segmentNucleiByQuarters) and returns the indices of the nuclei that overlap with the borderlines of the quarters
     and are therefore splitted in the segmentation results of segmentNucleiByQuarters

    Parameters
    ----------
    dapi_im: ndarray
        2D array containing data with float type
        Single channel grayscale Nuclei(DAPI) image

    nucModel: Instance of class MaskRCNN
        Mask RCNN model for Nuclei instance segmentation

    Returns
    -------
    mask_per_nucleus_in_border: ndarray
        Array with a 2D binary mask for every individual nucleus detected close to the borderline of the quarters
    splitted_nuclei_indices: ndarray
        1D array with boolean type
        indicates which of the binary masks in mask_per_nucleus_in_border belongs to a nucleus that is splitted in the
        segmentation results of segmentNucleiByQuarters (might be splitted into multiple nuclei)
    """
    # keep only the DAPI image that is in the proximity of the quarters borderlines
    # (100 pixels to from direction from the middle of the X and Y axis middle)
    boundary_area = dapi_im.copy()
    boundary_area[0: (IMAGE_HEIGHT/2)-100, 0: (IMAGE_WIDTH/2)-100] = 0
    boundary_area[0: (IMAGE_HEIGHT/2)-100, (IMAGE_WIDTH/2)+100:] = 0
    boundary_area[(IMAGE_HEIGHT/2)+100:, (IMAGE_WIDTH/2)+100:] = 0
    boundary_area[(IMAGE_HEIGHT/2)+100:, 0: (IMAGE_WIDTH/2)-100] = 0
    # Segment nuclei in the borderline proximity
    results = nucModel.detect([boundary_area], verbose=0)
    mask_per_nucleus_in_border = results[0]['masks']
    # detecting which nuclei is splitted
    splitted_nuclei_indices = getSplittedNucleiIndices(mask_per_nucleus_in_border)
    return mask_per_nucleus_in_border, splitted_nuclei_indices



def correctSplittedNuclei(mask_per_nucleus_in_border, splitted_nuclei_indices, nuclei_segmentation):
    """
    Corrects the instance segmentation based on segmentNucleiByQuarters so that nuclei that were partitioned due to the
    division to quarters will be correctly label into a single nucleus

    Parameters
    ----------
    mask_per_nucleus_in_border: ndarray
        Array with a 2D binary mask for every individual nucleus detected close to the borderline of the quarters
    splitted_nuclei_indices: ndarray
        1D array with boolean type
        indicates which of the binary masks in mask_per_nucleus_in_border belongs to a nucleus that is splitted in the
        segmentation results of segmentNucleiByQuarters (might be splitted into multiple nuclei)
    nuclei_segmentation: ndarray
        2D array containing data with int type
        Nuclei instance segmentation results for the entire DAPI image, each integer represents a different nucleus

    Returns
    -------
    nuclei_segmentation: ndarray
        2D array containing data with int type
        Nuclei instance segmentation results for the entire DAPI image after correction,
        each integer represents a different nucleus

    """
    splitted_nuclei_full_masks = mask_per_nucleus_in_border[:, :, splitted_nuclei_indices]
    for mask_idx in range(np.shape(splitted_nuclei_full_masks)[2]):
        # for each full nucleus detected in boundry area check to how many different nuclei it was partitioned to in the
        # nuclei_segmentation from segmentNucleiByQuarters
        full_nuc_mask = splitted_nuclei_full_masks[:, :, mask_idx]
        unique_nuclei = np.unique(nuclei_segmentation[full_nuc_mask])
        # the number of uniquely identified nuclei on the full nuclei area should not include background pixels with
        # value equal to 0
        num_unique_nuclei = len(unique_nuclei) if 0 not in unique_nuclei else len(unique_nuclei) - 1
        # remove partitioned nuclei and replace them with the full nucleus mask.
        if num_unique_nuclei > 1:
            min_label = np.inf
            for nuc_label in unique_nuclei:
                if nuc_label == 0:  # background is irrelevant
                    continue
                splitted_mask = nuclei_segmentation == nuc_label
                # remove the partitioned nucleus if it overlaps with the full nucleus area
                overlap = np.sum(np.logical_and(splitted_mask, full_nuc_mask)) / np.sum(splitted_mask)
                if overlap > 0.5:
                    nuclei_segmentation[splitted_mask] = 0
                    if nuc_label < min_label:
                        min_label = nuc_label
            # assign the full nucleus with the minimal label of its partitioned nuclei that it replaced
            nuclei_segmentation[full_nuc_mask] = min_label
    return nuclei_segmentation






def keepViableNuclei(nuclei_segmentation, nuclei_count, cells_foreground_mask):
    """
    Check for every nuclei in nuclei instance segmentation mask (nuclei_segmentation) if its viable by checking for
     overlap with the cell foreground mask

    Parameters
    ----------
    nuclei_segmentation:ndarray
        2D array containing data with int type
        Nuclei instance segmentation results for the entire DAPI image after correction,
        each integer represents a different nucleus
    nuclei_count: int
        Number of overall nuclei detected in the DAPI image
    cells_foreground_mask: ndarray
        2D array containing data with boolean type
        predicted cell foreground segmentation mask

    Returns
    -------
    nuclei_instance_segmentation_mask: ndarray
        2D array containing data with int type
        Final nuclei instance segmentation results for the entire DAPI image each integer represents a different nucleus
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers
    apoptosis_fraction: float
        fraction of non-viable cells in the field (nuclei that do not overlap with cell foreground)
    """
    centroids = [[0, 0]]
    # todo initialize centroids_new as empty array - this should include changes to other functions
    #initialize a new 2D array for segmentation results and a counter for viable cells (nuclei that overlap with foreground)
    nuclei_instance_segmentation_mask = np.zeros(np.shape(nuclei_segmentation))
    viable_cells_counter = 0
    # iterate other the possible nuclei labels: 0 is the background label,
    # and the maximal label possible is the nuclei_count which is the number
    # of nuclei before correcting partitioned nuclei
    for nuc_idx in range(1, nuclei_count):
        current_nuc_indices = nuclei_segmentation == nuc_idx
        if not np.any(current_nuc_indices):  # it might have been combined with another nucleus in correctSplittedNuclei
            continue
        overlap_with_foreground = np.mean(cells_foreground_mask[current_nuc_indices])
        if overlap_with_foreground > 0.7:
            # If the nucleus overlaps with foreground it is considered viable
            # and will appear in the instance segmenation results. In addition its pixels mean X,Y coordinates are saved
            # as its centroids
            viable_cells_counter += 1
            nuclei_instance_segmentation_mask[current_nuc_indices] = viable_cells_counter
            centroids.append(list(np.mean(np.where(current_nuc_indices), axis=1)))
    #todo denomenator in apoptosis_fraction should be changed to the amount of corrected nuclei after the splitting correction

    # the fraction of non-viable cells is is denoted as apoptosis_fraction
    apoptosis_fraction = 1 - viable_cells_counter / nuclei_count
    centroids = np.array(centroids).astype(int)
    return nuclei_instance_segmentation_mask, centroids, apoptosis_fraction



def segmentNuclei(dapi_im, cells_foreground_mask, nucModel):
    """
    Performs nuclei instance segmentation by dividing the image to 4 quarters, detecting the nuclei in each one of them,
    combine the results, correct partitioned nuclei due to the division and keep only the nuclei that are viable,
    meaning, they overlap with the cell foreground map. allows instance segmentation of about 4000 nuclei instead of
    1000 without the partition

    Parameters
    ----------
    dapi_im: ndarray
        2D array containing data with float type
        Single channel grayscale Nuclei(DAPI) image
    cells_foreground_mask: ndarray
        2D array containing data with boolean type
        predicted cell foreground segmentation mask
    nucModel: Instance of class MaskRCNN
        Mask RCNN model for Nuclei instance segmentation

    Returns
    -------
    nuclei_instance_segmentation_mask: ndarray
        2D array containing data with int type
        Final nuclei instance segmentation results for the entire DAPI image each integer represents a different nucleus
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers
    apoptosis_fraction: float
        fraction of non-viable cells in the field (nuclei that do not overlap with cell foreground)
    """
    # divide the DAPI image into quarters and perform instance segmentation on each of the quarters then
    # combine the results
    nuclei_segmentation, nuclei_count = segmentNucleiByQuarters(dapi_im, nucModel)
    # find nuclei in the proximity of the borderline of the quarters and check which one of them was partitioned
    # in segmentNucleiByQuarters
    mask_per_nucleus_in_border, splitted_nuclei_indices = findSplittedNuclei(dapi_im, nucModel)
    # if partitioned nuclei exist correct their segmentation
    if len(splitted_nuclei_indices) > 0:
        nuclei_segmentation = correctSplittedNuclei(mask_per_nucleus_in_border, splitted_nuclei_indices, nuclei_segmentation)
    # check which of the nuclei is the nucleus of a viable cell by checking overlap with cell foregroung mask
    nuclei_instance_segmentation_mask, centroids_new, apoptosis_fraction = keepViableNuclei(nuclei_segmentation, nuclei_count, cells_foreground_mask)
    return nuclei_instance_segmentation_mask, centroids_new, apoptosis_fraction



