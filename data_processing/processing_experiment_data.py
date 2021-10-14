import pickle
import os
import ast
from data_processing.outlier_removal import perform_full_outlier_removal
from data_processing.feature_extraction import *
from common import EPS
from sklearn.cluster import DBSCAN

def extract_saved_data(folder_path):
    """
    extracts the data from the copmuter vision pipeline output folder:
    pickle files names for each well and one txt file.

    Parameters
    ----------
    folder_path: str
        copmuter vision pipeline output folder

    Returns
    -------
    pkl_files_names: list
        list of the pickle files names in the output folder
    plate_data_txt: list
        lists of string that contain feature data from the computer vision pipeline (converted to dictionaries later)
    """
    pkl_files_names = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder_path) if '.txt' in name][0]
    plate_data_txt = [line.rstrip('\n') for line in open(os.path.join(folder_path, txt_file_name))]
    return pkl_files_names, plate_data_txt

def get_graph_per_field(pickle_file_full_path):
    well_graph_representation_dict = load_pickle(pickle_file_full_path)
    well_name = list(well_graph_representation_dict.keys())[0]
    graph_representation_per_field = well_graph_representation_dict[well_name]
    return graph_representation_per_field, well_name

def unpack_dictionary(line, well_name):
    d = ast.literal_eval(line)
    assert list(d.keys())[0] == well_name, 'well name not compatable'
    num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = d[well_name]["Cell Number"], d[well_name]["Neurite pixels"], d[well_name]["Apoptosis Fraction"], d[well_name]["Backscatter"], \
                                                                          d[well_name]["Neurite Distribution"]
    return num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist


def load_pickle(full_path):
    with open(full_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content

def calculate_well_features(graph_representation_per_field, well_data, conn_prob_density, thr_expected_conn):

    short_dst_count, mid_dst_count, long_dst_count, very_long_dst_count, cell_count, valid_fields_count = (0,)*6
    nnl, per_cell_neurite_length_distribution, ercr, dwn, edge_length_distribution = [], [], [], [], []

    for idx, graph_rep_list in enumerate(graph_representation_per_field):
        graph, node_dict = graph_rep_list
        num_cells = len(node_dict)
        cell_count += num_cells
        valid_fields_count += 1

        nnl.append((well_data["Neurite pixels"][idx] / num_cells))

        field_neu_dst = well_data["Neurite Distribution"][idx]
        per_cell_neurite_length_distribution += field_neu_dst
        field_neu_dst = np.array(field_neu_dst)

        per_cell_num_connections = np.array([d for n, d in graph.degree()])
        per_cell_expected_num_conn = ExpectedNumConnections(node_dict, conn_prob_density)
        ercr += list(per_cell_num_connections / (per_cell_expected_num_conn + 0.0001))

        dwn.append(pr_disconnected_with_neurite(per_cell_num_connections, field_neu_dst, per_cell_expected_num_conn, thr_expected_conn))

        field_edge_length_distribution = [weight for (n1, n2, weight) in graph.edges.data("weight")]

        edge_length_distribution += field_edge_length_distribution




        short_dist_pairs_count, mid_dist_pairs_count, long_dist_pairs_count, very_long_dist_pairs_count = count_pairs_in_distances(node_dict)
        short_dst_count += short_dist_pairs_count
        mid_dst_count += mid_dist_pairs_count
        long_dst_count += long_dist_pairs_count
        very_long_dst_count += very_long_dist_pairs_count

    short_edges, mid_edges, long_edges, very_long_edges = count_connections_in_distances(np.array(edge_length_distribution))

    return {"expectedVSreal_num_conn_s": np.nanmean(ercr),
                     "Valid Fields": valid_fields_count,
                     "neu2cells_ratio": np.nanmean(np.array(nnl)),
                     "Neurite Average": np.nanmean(per_cell_neurite_length_distribution),
                     "Disconnected With Neurites fd ref": np.nanmean(dwn),
                     "Short Connection Probability": short_edges / (short_dst_count + EPS),
                     "Intermediate Connection Probability": mid_edges / (mid_dst_count + EPS),
                     "Long Connection Probability": long_edges / (long_dst_count + EPS),
                     "Very Long Connection Probability": very_long_edges / (very_long_dst_count + EPS),
                     "# Cells": cell_count}

def calculate_well_features(graph_per_field, well_data, connection_pdf, thr_expected_conn):
    """
    Calculate the neurite outgrowth features from the well data.

    Parameters
    ----------
    graph_per_field: list
        a list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information
    well_data: dict
        A dictionary containing raw data from the computer vision pipeline
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)
    thr_expected_conn:float
        threshold for expected connections - cells with values lower than this are considered "isolated" in the
        calculation of "expected vs real connection ratio" feature.

    Returns
    -------
    feature_dict: dict
        A dictionary containing the features of the well
    """
    feature_dict = {"Expected VS Real Connection Ratio": [], "Valid Fields": 0, "Normalized Neurite Length": [], "Neurite Average": [], "Disconnected With Neurites": [], "Short Connection Probability": 0, "Intermediate Connection Probability": 0, "Long Connection Probability": 0, "Very Long Connection Probability": 0, "# Cells": 0}
    distances = ["Short", "Intermediate", "Long", "Very Long"]
    connection_prob_features_names = [n + " Connection Probability" for n in distances]
    cell_pairs_count_by_distance = {dst: 0 for dst in distances}
    edges_count_by_distance = {dst: 0 for dst in distances}
    well_edge_length_distribution = []

    for idx, graph_and_node_dict in enumerate(graph_per_field):
        graph, node_dict = graph_and_node_dict
        num_cells = len(node_dict)

        # update the number of cells in the field to the total number of cells in the well
        feature_dict["# Cells"] += num_cells

        # update the number of valid fields
        feature_dict["Valid Fields"] += 1

        # calculte the Normalized Neurite Length of the current well
        feature_dict["Normalized Neurite Length"].append((well_data["Neurite pixels"][idx] / num_cells))

        # update the well's neurites length distribution
        field_neurite_distribtuion = well_data["Neurite Distribution"][idx]
        feature_dict["Neurite Average"] += field_neurite_distribtuion

        # calculate the Expected VS Real Connection Ratio feature for current field
        per_cell_num_connections = np.array([d for n, d in graph.degree()])
        per_cell_expected_num_connections = ExpectedNumConnections(node_dict, connection_pdf)
        feature_dict["Expected VS Real Connection Ratio"] += list(per_cell_num_connections / (per_cell_expected_num_connections + 0.0001))

        # calculate the Disconnected With Neurites feature for current field
        dwn = pr_disconnected_with_neurite(per_cell_num_connections, np.array(field_neurite_distribtuion), per_cell_expected_num_connections, thr_expected_conn)
        feature_dict["Disconnected With Neurites"].append(dwn)

        # update the well's edge length distribution with the current field's distribution
        well_edge_length_distribution += [weight for (n1, n2, weight) in graph.edges.data("weight")]

        # update the count of cell pairs in specific distance ranges from each other
        for cell_pairs_count, dst in zip(count_pairs_in_distances(node_dict), distances):
            cell_pairs_count_by_distance[dst] += cell_pairs_count

    # count the edges in specific distance ranges
    for edges_count, dst in zip(count_connections_in_distances(np.array(well_edge_length_distribution)), distances):
        edges_count_by_distance[dst] = edges_count

    # calculate the well's probability of connection in a specific distance range as the ratio between the number of
    # connections made (edges) across this distance range (via neurites) to the amount of pairs of cells that are
    # in this distance range from one another (possible number of connections)
    for feature_name, dst in zip([connection_prob_features_names], distances):
        feature_dict[feature_name] = edges_count_by_distance[dst] / (cell_pairs_count_by_distance[dst] + EPS)

    # calculate the rest of the well wise features as the mean of the cell or field values
    for feature_name in feature_dict.keys():
        if (feature_name not in connection_prob_features_names) and (feature_name not in ["Valid Fields", "# Cells"]):
            feature_dict[feature_name] = np.nanmean(feature_dict[feature_name])

    return feature_dict


def calculate_connection_pdf(folder_path, negative_ref_wells):
    """
    Calculate the connection probability density function based on the negative reference wells.

    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    negative_ref_wells: list
        list containing the names (strings) of negative reference wells (disease model)

    Returns
    -------
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)
    """
    # extract the txt file and pickle files containing the output data of the computer vision pipeline
    pkl_files_names, plate_data_txt = extract_saved_data(folder_path)
    connection_prob_reference = []
    # iterate over each well's pickle file and txt data (line in the plate data txt file)
    for well_data_as_txt, pickle_file_full_path in zip(plate_data_txt, pkl_files_names):
        # extract the graph representation of this well's fields from the pickle file
        graph_per_field, well_name = get_graph_per_field(pickle_file_full_path)
        # extract data only from negative reference wells
        if well_name not in negative_ref_wells:
            continue
        # extract this well's data (dict) from the computer vision pipeline txt file
        well_data = ast.literal_eval(well_data_as_txt)[well_name]
        # perform outlier removal
        result_tuple, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field)
        # check that there are enough valid fields
        if len(result_tuple) == 1:
            continue

        # get the data of the fields that were left after outlier removal
        _, graph_per_field = result_tuple

        # iterate over the graph representations of fields from the current reference well and
        # calculate the probability of connection for each distance
        for idx, graph_and_node_dict in enumerate(graph_per_field):
            graph = graph_and_node_dict[0]
            node_dict = graph_and_node_dict[1]
            edges_length_list = [weight for (n1, n2, weight) in graph.edges.data("weight")]
            connection_pdf_field = calculate_connection_pdf4single_field(node_dict, np.array(edges_length_list))
            connection_prob_reference.append(list(connection_pdf_field))

    # connection pdf is defined as the mean (per distance) connection probability across the negative group
    connection_pdf = np.mean(connection_prob_reference, axis=0)
    return connection_pdf


def calculate_isolated_cells_threshold(folder_path, negative_ref_wells, connection_pdf):
    """
    Calculate the threshold for isolated cells by calculating the distribution of expected connections (per cell) of the
    negative reference, based on the connection probability density function. Threshold is calculated as the
    10th percentile of the distribution.
    
    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    negative_ref_wells: list
        list containing the names (strings) of negative reference wells (disease model)
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)

    Returns
    -------
    threshold_expected_connection: float
        Threshold for isolated cells (cells with lower expected connections are considered isolated)
    """
    # extract the txt file and pickle files containing the output data of the computer vision pipeline
    pkl_files_names, plate_data_txt = extract_saved_data(folder_path)
    expected_conn = []
    # iterate over each well's pickle file and txt data (line in the plate data txt file)
    for well_data_as_txt, pickle_file_full_path in zip(plate_data_txt, pkl_files_names):
        # extract the graph representation of this well's fields from the pickle file
        graph_per_field, well_name = get_graph_per_field(pickle_file_full_path)
        # extract data only from negative reference wells
        if well_name not in negative_ref_wells:
            continue
        # extract this well's data (dict) from the computer vision pipeline txt file
        well_data = ast.literal_eval(well_data_as_txt)[well_name]
        # perform outlier removal
        result_tuple, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field)
        # check that there are enough valid fields
        if len(result_tuple) == 1:
            continue
        # get the data of the fields that were left after outlier removal
        _, graph_per_field = result_tuple

        # calculate the expected number of connections for each cell in each field in the well
        for idx, graph_and_node_dict in enumerate(graph_per_field):
            node_dict = graph_and_node_dict[1]
            expected_conn += list(ExpectedNumConnections(node_dict, connection_pdf))

    # set the threshold for an isolated cell at the 10th percentile
    threshold_expected_connection = np.percentile(expected_conn, 10)
    return threshold_expected_connection


def calculate_plate_outgrowth_measures(folder_path, connection_pdf, thr_expected_conn):
    """
    Calculates neurite-outgrowth and toxicity measures
    for each well from the data extracted in the computer vision pipeline.

    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)
    thr_expected_conn: float
        threshold for expected connections - cells with values lower than this are considered "isolated" in the
        calculation of "expected vs real connection ratio" feature.

    Returns
    -------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    """
    # extract the txt file and pickle files containing the output data of the computer vision pipeline
    pkl_files_names, plate_data_txt = extract_saved_data(folder_path)
    plate_processed_data = {}
    # iterate over each well's pickle file and txt data (line in the plate data txt file)
    for well_data_as_txt, pickle_file_full_path in zip(plate_data_txt, pkl_files_names):
        # extract the graph representation of this well's fields from the pickle file
        graph_per_field, well_name = get_graph_per_field(pickle_file_full_path)
        # extract this well's data (dict) from the computer vision pipeline txt file
        well_data = ast.literal_eval(well_data_as_txt)[well_name]
        # perform outlier removal
        result_tuple, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field)
        # check that there are enough valid fields
        if len(result_tuple) == 1:
            plate_processed_data[well_name] = {"outlier_dictionary": outlier_dict}
            continue
        # get the data of the fields that were left after outlier removal
        well_data, graph_per_field = result_tuple
        # calculate the neurite outgrowth features
        well_features = calculate_well_features(graph_per_field, well_data, connection_pdf, thr_expected_conn)
        well_features["outlier_dictionary"] = outlier_dict
        plate_processed_data[well_name] = well_features
    return plate_processed_data


def process_plate_data(folder_path, negative_ref_wells):
    """
    Process plate data extracted from the computer vision pipeline. processing includes extracting
    a connection probability function and thresholds from the negative reference wells followed by the
    extraction of neurite-outgrowth, toxicity and outlier data from each well in the plate.

    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    negative_ref_wells: list
        list containing the names (strings) of negative reference wells (disease model)

    Returns
    -------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    """
    connection_pdf = calculate_connection_pdf(folder_path, negative_ref_wells)
    # calculate expected connection thresholds
    thr_expected_connection = calculate_isolated_cells_threshold(folder_path, negative_ref_wells, connection_pdf)
    # extract neurite outgrowth measures and outlier information from each well in the plate
    plate_processed_data = calculate_plate_outgrowth_measures(folder_path, connection_pdf, thr_expected_connection)
    return plate_processed_data
