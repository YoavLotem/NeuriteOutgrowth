import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.stats import norm
from itertools import compress
from common import MAX_APOP_RATIO, MAX_CELL_NUM, MIN_CELL_NUM, MAX_HIGH_DENSITY_RATIO, DB, MIN_VALID_FIELDS
from sklearn.cluster import DBSCAN


def detect_outliers_with_thresholds(well_data, list_of_graph_embedings):
    filter_mask = np.zeros_like(well_data["Cell Number"], dtype=bool)
    apop_fields, low_cell_fields, high_cell_fields, clustered_cells = 0, 0, 0, 0
    for idx, graph_and_node_list in enumerate(list_of_graph_embedings):
        node_list = graph_and_node_list[1]
        num_cells = len(node_list)
        if num_cells < MIN_CELL_NUM:
            low_cell_fields += 1
            continue
        if num_cells > MAX_CELL_NUM:
            high_cell_fields += 1
            continue
        if well_data["Apoptosis Fraction"][idx] > MAX_APOP_RATIO:
            apop_fields += 1
            continue
        X = np.array(list(node_list.values()))
        fitted_db = DB.fit(X)
        density_feat = len(fitted_db.core_sample_indices_) / len(X)
        if density_feat > MAX_HIGH_DENSITY_RATIO:
            clustered_cells += 1
            continue
        filter_mask[idx] = True
    outlier_dict = {"Number of Fields With Low Cell Count": low_cell_fields, "Number of Fields With High Cell Count": high_cell_fields, "Number of Fields With Apoptosis": apop_fields, "Apoptosis Ratio Before Outlier Removal": round(np.mean(well_data["Apoptosis Fraction"]), 2), "Number of Fields With Clustered Cells": clustered_cells}
    return filter_mask, outlier_dict


def filter_outliers_ransac(well_data):
    num_cells = np.array(well_data["Cell Number"])
    normalized_neurite_length = np.array(well_data["Neurite pixels"]) / num_cells
    X = num_cells.reshape(-1, 1)
    y = normalized_neurite_length.reshape(-1, 1)
    std = np.std(y)
    ransac = RANSACRegressor(residual_threshold=1 * std, min_samples=5, random_state=42)
    ransac.fit(X, y)
    yhat = ransac.predict(X)
    R = y - yhat
    mu, std = norm.fit(R)
    n = norm(mu, std)
    inliers = n.cdf(R) > 0.05
    return inliers[:, 0]


def filter_data(graph_representation_per_field, well_data, filter_mask):
    graph_representation_per_field = list(compress(graph_representation_per_field, filter_mask))
    well_data_filtered = {feature: list(compress(well_data[feature], filter_mask)) for feature in well_data.keys()}
    return graph_representation_per_field, well_data_filtered


def perform_full_outlier_removal(well_data, graph_representation_per_field):
    # remove outliers with thresholds
    filter_mask, outlier_dict = detect_outliers_with_thresholds(well_data, graph_representation_per_field)
    # filter data to remove outliers
    graph_representation_per_field, well_data = filter_data(graph_representation_per_field, well_data, filter_mask)

    if np.sum(filter_mask) < MIN_VALID_FIELDS:
        # print(well_name + ' has under 5 valid fields after initial filtering')
        outlier_dict["Valid Fields"] = np.sum(filter_mask)
        outlier_dict["Number of RANSAC Outlier Fields"] = 0
        outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(well_data["Apoptosis Fraction"]), 2)
        return [False], outlier_dict

    # remove outliers with RANSAC
    filter_mask_ransac = filter_outliers_ransac(well_data)
    graph_representation_per_field, well_data = filter_data(graph_representation_per_field, well_data, filter_mask_ransac)
    outlier_dict["Number of RANSAC Outlier Fields"] = np.sum(filter_mask) - np.sum(filter_mask_ransac)
    outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(well_data["Apoptosis Fraction"]), 2)


    if np.sum(filter_mask_ransac) < MIN_VALID_FIELDS:
        # print(well_name + ' has under 5 valid fields after Ransac filtering')
        outlier_dict["Valid Fields"] = np.sum(filter_mask_ransac)
        return [False], outlier_dict

    return (well_data, graph_representation_per_field), outlier_dict
