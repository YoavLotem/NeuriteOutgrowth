import numpy as np
import scipy
from common import MIN_DISTANCE, MAX_DISTANCE, BIN_SIZE


def calculate_expected_number_of_connections(node_dict, connection_pdf):
    node_arr = np.array(list(node_dict.values()))
    pairwise_distances_condensed = scipy.spatial.distance.pdist(node_arr)
    pairwise_distances_squareform = scipy.spatial.distance.squareform(pairwise_distances_condensed)
    per_cell_expected_num_connections = np.zeros(len(node_arr))
    distances_list = np.arange(MIN_DISTANCE, MAX_DISTANCE, BIN_SIZE)
    for index_of_bin_in_pdf, distance in enumerate(distances_list):
        neighbours_in_distance = np.sum(np.logical_and(pairwise_distances_squareform < (distance + BIN_SIZE),
                                                       pairwise_distances_squareform >= distance), axis=1)
        per_cell_expected_num_connections += neighbours_in_distance * connection_pdf[index_of_bin_in_pdf]
    return per_cell_expected_num_connections


def count_pairs(node_dict):
    node_arr = np.array(list(node_dict.values()))
    pairwise_distances_condensed = scipy.spatial.distance.pdist(node_arr)
    short_dist = np.sum(pairwise_distances_condensed <= 100)
    mid_dist = np.sum(np.logical_and(pairwise_distances_condensed > 100, pairwise_distances_condensed <= 300))
    long_dist = np.sum(np.logical_and(pairwise_distances_condensed > 300, pairwise_distances_condensed <= 400))
    very_long = np.sum(pairwise_distances_condensed > 300)
    return short_dist, mid_dist, long_dist, very_long


def count_connections(edge_lenghts):
    short_edges = np.sum(edge_lenghts <= 100)
    mid_edges = np.sum(np.logical_and(edge_lenghts > 100, edge_lenghts <= 300))
    long_edges = np.sum(np.logical_and(edge_lenghts > 300, edge_lenghts <= 400))
    very_long_edges = np.sum(edge_lenghts > 300)
    return short_edges, mid_edges, long_edges, very_long_edges



def pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn, threshold_expected_connection):
    pr_disconnected_with_neu_not_isolated = np.sum(np.logical_and(np.logical_and(degrees == 0, temp_image_neu_dst != 0), expected_num_conn > threshold_expected_connection))
    pr_with_neu_not_isolated = np.sum(np.logical_and(temp_image_neu_dst != 0, expected_num_conn > threshold_expected_connection))
    conditional = pr_disconnected_with_neu_not_isolated / (pr_with_neu_not_isolated + 0.00001)
    return (len(degrees) ** 0.5) * conditional


def calculate_connection_pdf_for_a_single_field(node_list, edge_lenghts):
    node_arr = np.array(list(node_list.values()))
    d1 = scipy.spatial.distance.pdist(node_arr)
    prob_arr = np.zeros((41))
    distances = np.arange(25, 1050, 25)
    deno0 = np.sum(d1 <= 25)
    numer0 = np.sum(edge_lenghts <= 0)
    if deno0 == 0:
        prob_arr[0] = 0
    else:
        prob_arr[0] = numer0 / deno0

    for i, dist in enumerate(distances):
        deno = np.sum(np.logical_and(d1 <= dist, d1 > dist - 25))
        numer = np.sum(np.logical_and(edge_lenghts <= dist, edge_lenghts > dist - 25))
        if deno == 0:
            prob_arr[i] = 0
        else:
            prob_arr[i] = numer / deno
    return prob_arr