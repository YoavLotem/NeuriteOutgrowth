import ast
from src.data_processing.outlier_removal import perform_full_outlier_removal
from src.data_processing.feature_extraction import *
from src.data_processing.utils import *

def calculate_well_features(graph_per_field, well_data, connection_pdf, thr_expected_conn, exp_config):
    """
    Calculate the neurite outgrowth features from the well data.

    Parameters
    ----------
    graph_per_field: list
        A list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information
    well_data: dict
        A dictionary containing a well's raw data from the computer vision pipeline
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)
    thr_expected_conn:float
        threshold for expected connections - cells with values lower than this are considered "isolated" in the
        calculation of "expected vs real connection ratio" feature.
    exp_config: Instance of class ExperimentConfig
        Holds many tune-able parameters of the experiment (thresholds for outlier removal etc.)
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
        field_neurite_distribution = well_data["Neurite Distribution"][idx]
        feature_dict["Neurite Average"] += field_neurite_distribution

        # calculate the Expected VS Real Connection Ratio feature for current field
        per_cell_num_connections = np.array([d for n, d in graph.degree()])
        per_cell_expected_num_connections = calculate_expected_number_of_connections(node_dict, connection_pdf, exp_config)
        feature_dict["Expected VS Real Connection Ratio"] += list(per_cell_num_connections / (per_cell_expected_num_connections + 0.0001))

        # calculate the Disconnected With Neurites feature for current field
        dwn = calculate_disconnected_with_neurites(per_cell_num_connections, np.array(field_neurite_distribution), per_cell_expected_num_connections, thr_expected_conn)

        feature_dict["Disconnected With Neurites"].append(dwn)

        # update the well's edge length distribution with the current field's distribution
        well_edge_length_distribution += [weight for (n1, n2, weight) in graph.edges.data("weight")]

        # update the count of cell pairs in specific distance ranges from each other
        for cell_pairs_count, dst in zip(count_pairs(node_dict), distances):
            cell_pairs_count_by_distance[dst] += cell_pairs_count

    # count the edges in specific distance ranges
    for edges_count, dst in zip(count_connections(np.array(well_edge_length_distribution)), distances):
        edges_count_by_distance[dst] = edges_count

    # calculate the well's probability of connection in a specific distance range as the ratio between the number of
    # connections made (edges) across this distance range (via neurites) to the amount of pairs of cells that are
    # in this distance range from one another (possible number of connections)
    for feature_name, dst in zip(connection_prob_features_names, distances):
        feature_dict[feature_name] = edges_count_by_distance[dst] / (cell_pairs_count_by_distance[dst] + exp_config.EPS)

    # calculate the rest of the well wise features as the mean of the cell or field values
    for feature_name in feature_dict.keys():
        if (feature_name not in connection_prob_features_names) and (feature_name not in ["Valid Fields", "# Cells"]):
            feature_dict[feature_name] = np.nanmean(feature_dict[feature_name])

    return feature_dict


def calculate_connection_pdf(folder_path, negative_ref_wells, exp_config):
    """
    Calculate the connection probability density function - the probability of connection over multiple distance ranges,
    (25 [pixels] bins covering the 0-1000 [pixels] range) based on the negative reference wells.

    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    negative_ref_wells: list
        list containing the names (strings) of negative reference wells (disease model)
    exp_config: Instance of class ExperimentConfig
        Holds many tune-able parameters of the experiment (thresholds for outlier removal etc.)
    Returns
    -------
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance range
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
        well_data, graph_per_field, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field, exp_config)
        # check that there are enough valid fields
        if outlier_dict["Valid Fields"] < exp_config.MIN_VALID_FIELDS:
            continue

        # iterate over the graph representations of fields from the current reference well and
        # calculate the probability of connection for each distance
        for idx, graph_and_node_dict in enumerate(graph_per_field):
            graph = graph_and_node_dict[0]
            node_dict = graph_and_node_dict[1]
            edges_length_list = [weight for (n1, n2, weight) in graph.edges.data("weight")]
            connection_pdf_field = calculate_connection_pdf_for_a_single_field(node_dict, np.array(edges_length_list), exp_config)
            connection_prob_reference.append(list(connection_pdf_field))

    # connection pdf is defined as the mean (per distance) connection probability across the negative group
    connection_pdf = np.mean(connection_prob_reference, axis=0)
    return connection_pdf


def calculate_isolated_cells_threshold(folder_path, negative_ref_wells, connection_pdf, exp_config):
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
    exp_config: Instance of class ExperimentConfig
        Holds many tune-able parameters of the experiment (thresholds for outlier removal etc.)
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
        well_data, graph_per_field, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field, exp_config)
        # check that there are enough valid fields
        if outlier_dict["Valid Fields"] < exp_config.MIN_VALID_FIELDS:
            continue

        # calculate the expected number of connections for each cell in each field in the well
        for idx, graph_and_node_dict in enumerate(graph_per_field):
            node_dict = graph_and_node_dict[1]
            expected_conn += list(calculate_expected_number_of_connections(node_dict, connection_pdf, exp_config))

    # set the threshold for an isolated cell at the 10th percentile
    threshold_expected_connection = np.percentile(expected_conn, 20)
    return threshold_expected_connection


def calculate_plate_outgrowth_measures(folder_path, connection_pdf, thr_expected_conn, exp_config):
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
    exp_config: Instance of class ExperimentConfig
        Holds many tune-able parameters of the experiment (thresholds for outlier removal etc.)
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
        well_data, graph_per_field, outlier_dict = perform_full_outlier_removal(well_data, graph_per_field, exp_config)
        # check that there are enough valid fields
        if outlier_dict["Valid Fields"] < exp_config.MIN_VALID_FIELDS:
            plate_processed_data[well_name] = {"outlier_dictionary": outlier_dict}
            continue
        # calculate the neurite outgrowth features
        well_features = calculate_well_features(graph_per_field, well_data, connection_pdf, thr_expected_conn, exp_config)
        well_features["outlier_dictionary"] = outlier_dict
        plate_processed_data[well_name] = well_features
    return plate_processed_data


def process_plate_data(folder_path, negative_ref_wells, exp_config):
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
    exp_config: Instance of class ExperimentConfig
        Holds many tune-able parameters of the experiment (thresholds for outlier removal etc.)
    Returns
    -------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    """
    connection_pdf = calculate_connection_pdf(folder_path, negative_ref_wells, exp_config)
    # calculate expected connection thresholds
    thr_expected_connection = calculate_isolated_cells_threshold(folder_path, negative_ref_wells, connection_pdf, exp_config)
    # extract neurite outgrowth measures and outlier information from each well in the plate
    plate_processed_data = calculate_plate_outgrowth_measures(folder_path, connection_pdf, thr_expected_connection, exp_config)
    return plate_processed_data
