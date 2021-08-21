import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.stats import norm
from itertools import compress


def filter_outliers(num_cells_list, apop_list, neu_pix_list, db, list_of_graph_embedings, well_name):
    filter_mask = np.zeros_like(num_cells_list, dtype=bool)
    apop_fields, low_cell_fields, high_cell_fields, clustered_cells = 0, 0, 0, 0
    for idx, graph_and_node_list in enumerate(list_of_graph_embedings):
        node_list = graph_and_node_list[1]
        num_cells = len(node_list)
        if num_cells < 50:
            low_cell_fields += 1
            continue
        if num_cells > 1000:
            high_cell_fields += 1
            continue
        if apop_list[idx] > 0.25:
            apop_fields += 1
            continue
        X = np.array(list(node_list.values()))
        fitted_db = db.fit(X)
        density_feat = len(fitted_db.core_sample_indices_) / len(X)
        if density_feat > 0.45:
            clustered_cells += 1
            continue
        filter_mask[idx] = True
    outlier_dict = {"Number of Fields With under 50 Cells": low_cell_fields, "Number of Fields With over 1000 Cells": high_cell_fields, "Number of Fields With Apoptosis": apop_fields, "Apoptosis Ratio Before Outlier Removal": round(np.mean(apop_list), 2), "Number of Fields With Clustered Cells": clustered_cells}
    return filter_mask, outlier_dict


def filter_outliers_ransac(num_cells, Normalized_Neurite_Length):
    X = num_cells.reshape(-1, 1)
    y = Normalized_Neurite_Length.reshape(-1, 1)
    std = np.std(y)
    ransac = RANSACRegressor(residual_threshold=1 * std, min_samples=5)
    ransac.fit(X, y)
    yhat = ransac.predict(X)
    R = y - yhat
    mu, std = norm.fit(R)
    n = norm(mu, std)
    inliers = n.cdf(R) > 0.05
    return inliers[:, 0]




def total_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name):
    # remove outliers with thresholds
    filter_mask, outlier_dict = filter_outliers(num_cells_list, apop_list, neu_pix_list, db, list_of_graph_embedings, well_name)
    num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = [list(compress(data, filter_mask)) for data in
                                                                                                    [num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings]]
    if np.sum(filter_mask) < 5:
        # print(well_name + ' has under 5 valid fields after initial filtering')
        outlier_dict["Valid Fields"] = np.sum(filter_mask)
        outlier_dict["Number of RANSAC Outlier Fields"] = 0
        outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(apop_list), 2)
        return [False], outlier_dict

    # remove outliers with RANSAC
    num_cells = np.array(num_cells_list)
    neu_pix = np.array(neu_pix_list)
    Normalized_Neurite_Length = neu_pix / num_cells
    filter_mask_ransac = filter_outliers_ransac(num_cells, Normalized_Neurite_Length)
    num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = [list(compress(data, filter_mask_ransac)) for data in [num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings]]
    outlier_dict["Number of RANSAC Outlier Fields"] = np.sum(filter_mask) - np.sum(filter_mask_ransac)
    outlier_dict["Apoptosis Ratio After Outlier Removal"] = round(np.mean(apop_list), 2)
    if np.sum(filter_mask_ransac) < 5:
        # print(well_name + ' has under 5 valid fields after Ransac filtering')
        outlier_dict["Valid Fields"] = np.sum(filter_mask_ransac)
        return [False], outlier_dict

    return (num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings), outlier_dict
