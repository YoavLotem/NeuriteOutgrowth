from sklearn.linear_model import RANSACRegressor
from scipy.stats import norm
from itertools import compress
from src.common import *
from sklearn.cluster import DBSCAN


def detect_outliers_with_thresholds(well_data, graph_per_field):
    """
    Detect fields that do not pass threshold regarding the number of cells, levels of apoptosis,
    and density of their cells.
    
    Parameters
    ----------
    well_data: dict
        A dictionary containing a well's raw data from the computer vision pipeline
    graph_per_field: list
        A list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information

    Returns
    -------
    inlier_mask: ndarray
        1D ndarray of type boolean to indicate inlier fields to keep
    outlier_dict: dict
        dictionary containing outlier and apoptosis data of the current well
    """
    # initializing a boolean mask to indicate which field to keep after outlier removal
    inlier_mask = np.zeros_like(well_data["Cell Number"], dtype=bool)
    # initializing a dictionary to hold outlier and apoptosis data
    outlier_dict = {"Number of Fields With Low Cell Count": 0,
                    "Number of Fields With High Cell Count": 0,
                    "Number of Fields With Apoptosis": 0,
                    "Number of Fields With Clustered Cells": 0,
                    "Apoptosis Ratio Before Outlier Removal": round(np.mean(well_data["Apoptosis Fraction"]), 2)}

    # iterating over the field's data - only fields that passes all threshold is considered an inlier.
    for idx, graph_and_node_list in enumerate(graph_per_field):
        node_list = graph_and_node_list[1]
        num_cells = len(node_list)
        if num_cells < MIN_CELL_NUM:
            outlier_dict["Number of Fields With Low Cell Count"] += 1
            continue
        if num_cells > MAX_CELL_NUM:
            outlier_dict["Number of Fields With High Cell Count"] += 1
            continue
        if well_data["Apoptosis Fraction"][idx] > MAX_APOP_RATIO:
            outlier_dict["Number of Fields With Apoptosis"] += 1
            continue

        # using the DBSCAN algorithm's core samples as highly dense cells (check algorithm for details)
        cell_coordinates = np.array(list(node_list.values()))
        db = DBSCAN(eps=D_EPS, min_samples=MIN_SAMPLES)
        fitted_db = db.fit(cell_coordinates)
        density_feat = len(fitted_db.core_sample_indices_) / len(cell_coordinates)
        if density_feat > MAX_HIGH_DENSITY_RATIO:
            outlier_dict["Number of Fields With Clustered Cells"] += 1
            continue

        # a field that passed all threshold is an inlier
        inlier_mask[idx] = True
    return inlier_mask, outlier_dict


def detect_outliers_unsupervised(well_data):
    """
    Detect fields with extreme values compared to the other fields in the well using an unsupervised approach based on
    the RANSAC regressor.

    Parameters
    ----------
    well_data: dict
        A dictionary containing a well's raw data from the computer vision pipeline

    Returns
    -------
    inlier_mask: ndarray
        1D ndarray of type boolean to indicate inlier fields to keep
    """
    # set X-axis variable as number of cells and Y-axis variable as normalized neurite length (NNL)
    num_cells = np.array(well_data["Cell Number"])
    normalized_neurite_length = np.array(well_data["Neurite pixels"]) / num_cells
    x = num_cells.reshape(-1, 1)
    y = normalized_neurite_length.reshape(-1, 1)
    nnl_std = np.std(y)
    # fit a RANSAC regressor using 1 NNL standard deviation as residual threshold
    ransac = RANSACRegressor(residual_threshold=nnl_std, min_samples=RANSAC_MIN_SAMPLES, random_state=42)
    ransac.fit(x, y)
    # calculate the residual distance between the predicted NNL by the RANSAC regressor (prd) and the observed NNL (y)
    prd = ransac.predict(x)
    residual_distance = y - prd
    # fit a normal distribution to the residual distance distribution
    mu, std = norm.fit(residual_distance)
    residual_distance_normal_distribution = norm(mu, std)
    # inliers are the samples that have cdf > threshold -> meaning their values are not extremely far away than the
    # residual line calculated by the RANSAC regressor
    inlier_mask = residual_distance_normal_distribution.cdf(residual_distance) > PROBABILITY_THRESHOLD
    inlier_mask = inlier_mask[:, 0]
    return inlier_mask


def filter_data(graph_per_field, well_data, inlier_mask):
    """
    
    Parameters
    ----------
    graph_per_field: list
        A list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information
    well_data: dict
        A dictionary containing a well's raw data from the computer vision pipeline
    inlier_mask: ndarray
        1D ndarray of type boolean to indicate inlier fields to keep

    Returns
    -------
    graph_per_field_filtered: list
        same as input graph_per_field but with outliers fields' data removed from list
    well_data_filtered: dict
        same as input well_data but with outliers fields' data removed from every list in the dictionary
    """
    # filter graph_per field using the inlier_mask to indicate which field's data to keep
    graph_per_field_filtered = list(compress(graph_per_field, inlier_mask))
    # filter every list in the well_data dictionary using the inlier_mask to indicate which field's data to keep
    well_data_filtered = {feature: list(compress(well_data[feature], inlier_mask)) for feature in well_data.keys()}
    return graph_per_field_filtered, well_data_filtered


def perform_full_outlier_removal(well_data, graph_per_field):
    """
    Performs the full outlier removal process, which includes detecting and removing outliers using constant thresholds
    and using an unsupervised approach of removing fields with extreme values.

    Parameters
    ----------
    well_data: dict
        A dictionary containing a well's raw data from the computer vision pipeline
    graph_per_field: list
        A list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information

    Returns
    -------
    well data: dict
        same as input well_data but with outliers fields' data removed from every list in the dictionary
    graph_per_field: list
        same as input graph_per_field but with outliers fields' data removed from list
    outlier_dict: dict
        dictionary containing outlier and apoptosis data of the current well

    """
    # detect and remove outliers with constant thresholds
    inlier_mask, outlier_dict = detect_outliers_with_thresholds(well_data, graph_per_field)
    graph_per_field, well_data = filter_data(graph_per_field, well_data, inlier_mask)

    # if the number of valid fields (inliers) is smaller than threshold than we do not continue with outlier removal
    # and the well's results should be considered invalid
    if np.sum(inlier_mask) < MIN_VALID_FIELDS:
        outlier_dict["Valid Fields"] = np.sum(inlier_mask)
        outlier_dict["Number of Unsupervised Outlier Fields"] = 0
        outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(well_data["Apoptosis Fraction"]), 2)
        return well_data, graph_per_field, outlier_dict

    # detect and remove outliers compared to the other fields' values
    inlier_mask_unsupervised = detect_outliers_unsupervised(well_data)
    graph_per_field, well_data = filter_data(graph_per_field, well_data, inlier_mask_unsupervised)
    # the number of outliers detected in this stage is the difference between the inliers in the previous stage
    # and the inliers after the current stage
    outlier_dict["Number of Unsupervised Outlier Fields"] = np.sum(inlier_mask) - np.sum(inlier_mask_unsupervised)
    # save the apoptosis ratio after the total outlier removal
    outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(well_data["Apoptosis Fraction"]), 2)

    # the amount of valid fields will indiate whether the well is valid and its data reliable
    outlier_dict["Valid Fields"] = np.sum(inlier_mask_unsupervised)

    return well_data, graph_per_field, outlier_dict
