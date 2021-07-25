from sklearn.mixture import GaussianMixture
import cv2
import numpy as np

def postProcessing(mask, thr=100):
    """
    This function performs post-processing for a binary segmentation map by
    removing connected components whose size [pixels] is smaller than a provided threshold
    :param mask: Binary segmentation map before pre-processing (numpy array)
    :param thr: Threshold (int) that indicates the minimum size of a connected
     component that will be kept
    :return: Binary segmentation map after removing small connected components (numpy array)
    """
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    mask2 = np.zeros((2048, 2048))

    # cleaning the image from small particles
    for i in range(0, nb_components):
        if sizes[i] >= thr:
            mask2[output == i + 1] = 1
    return mask2

def findneu(image, neurite_model):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image - np.mean(image)
    image = image / 255
    X = image[np.newaxis, :, :, np.newaxis]
    prd = neurite_model.predict(X, steps=1)
    prd = prd[0, :, :, 0]
    prd = prd > 0.5
    mask = postProcessing(prd)
    return mask

def foreground(FITC):
    FITC2 = cv2.resize(FITC, (512, 512), interpolation=cv2.INTER_AREA)
    classif = GaussianMixture(n_components=2)
    classif.fit(FITC2.reshape((FITC2.size, 1)))
    threshold = np.mean(classif.means_)
    cells = FITC > threshold
    return cells


def create_nuc_im(dapi_im, segmented_image, nucModel):
    # detecting the nuclei using mask rcnn (divide the image to 4 for case of over 1000 cells)
    nuc_markers_total = np.zeros((2048, 2048))
    counter = 1
    for i in range(2):
        for j in range(2):
            quarter_dapi = dapi_im[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024, :]
            results = nucModel.detect([quarter_dapi], verbose=0)
            m = results[0]['masks']
            numnucs = np.shape(m)[2]
            nuc_markers = np.zeros((1024, 1024))
            for idx in range(numnucs):
                mask = m[:, :, idx]
                nuc_markers[mask] = counter
                counter += 1
            nuc_markers_total[i * 1024:(i + 1) * 1024, j * 1024:(j + 1) * 1024] = nuc_markers

    # detect nucleuses that might have been splitted in the middle by partition to quarters
    splitting_area = dapi_im.copy()
    # detecting nucs only in the proximity of the splitting line
    splitting_area[0:924, 0:924] = 0
    splitting_area[1124:, 1124:] = 0
    splitting_area[0:924, 1124:] = 0
    splitting_area[1124:, 0:924] = 0
    results = nucModel.detect([splitting_area], verbose=0)
    r = results[0]
    m = r['masks']

    # detecting which nucleuses are potentially splitted
    cross = np.zeros((np.shape(m)))
    cross[1023:1025, :, :] = 1
    cross[:, 1023:1025, :] = 1
    multi = np.any(np.logical_and(cross, m), (0, 1))
    splitted = m[:, :, multi]
    # changing those splitted nucs to a single nuc
    for mask_idx in range(np.shape(splitted)[2]):
        full_nuc_mask = splitted[:, :, mask_idx]
        unique_nucs = np.unique(nuc_markers_total[full_nuc_mask])
        num_unique_nucs = len(unique_nucs) if 0 not in unique_nucs else len(unique_nucs) - 1
        if num_unique_nucs > 1:
            # removing old nucs if they overlap significantly with the complete nuc
            min_num = np.inf
            for nuc in unique_nucs:
                if nuc == 0:
                    continue
                splitted_mask = nuc_markers_total == nuc
                overlap = np.sum(np.logical_and(splitted_mask, full_nuc_mask)) / np.sum(splitted_mask)
                if overlap > 0.5:
                    nuc_markers_total[splitted_mask] = 0
                    if nuc < min_num:
                        min_num = nuc

            nuc_markers_total[full_nuc_mask] = min_num

    centroids_new = [[0, 0]]
    # removing nuclueses without 70% overlap with cells foreground
    nuc_im = np.zeros(np.shape(nuc_markers_total))
    good_cells_counter = 0
    # map between values in nuc_markers_total and in nuc_im for splitted cells
    good_cell_map = {}
    for nuc_idx in range(1, counter):
        current_nuc_indices = nuc_markers_total == nuc_idx
        if not np.any(current_nuc_indices):
            continue
        overlap_with_foreground = np.mean(segmented_image[current_nuc_indices])
        if overlap_with_foreground > 0.7:
            good_cells_counter += 1
            nuc_im[current_nuc_indices] = good_cells_counter
            centroids_new.append(list(np.mean(np.where(current_nuc_indices), axis=1)))
    apoptosis_fraction = 1 - good_cells_counter / counter

    return nuc_im, np.array(centroids_new).astype(int), apoptosis_fraction


