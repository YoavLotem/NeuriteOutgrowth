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


def calculate_connection_pdf(folder_path, ref_names, db):
    pkl_files_names, plate_data_txt = extract_saved_data(folder_path)
    prob_ref = []
    for line, file_full_path in zip(plate_data_txt, pkl_files_names):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        if well_name not in ref_names:
            continue
        list_of_graph_embedings = well_dictionary[well_name]
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)

        result_tupple, _ = perform_full_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)
        if len(result_tupple) == 1:
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = result_tupple
        for idx, graph_and_node_list in enumerate(list_of_graph_embedings):
            graph = graph_and_node_list[0]
            node_list = graph_and_node_list[1]
            length_list = [weight for (n1, n2, weight) in graph.edges.data("weight")]
            prob_arr = connectivity_per_distance_pdf(node_list, np.array(length_list))
            prob_ref.append(list(prob_arr))
    if not prob_ref:
        return False
    connection_probability_density = np.mean(prob_ref, axis=0)
    return connection_probability_density









def calculate_thresholds(folder_path, ref_names, db, prob):
    pkl_files_names, plate_data_txt = extract_saved_data(folder_path)
    expected_conn = []
    for line, file_full_path in zip(plate_data_txt, pkl_files_names):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        if well_name not in ref_names:
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)
        list_of_graph_embedings = well_dictionary[well_name]

        result_tupple, _ = perform_full_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)
        if len(result_tupple) == 1:
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = result_tupple

        for idx, graph_and_node_list in enumerate(list_of_graph_embedings):
            node_list = graph_and_node_list[1]
            expected_conn += list(ExpectedNumConnections(node_list, prob))

    threshold_expected_connection = np.percentile(expected_conn, 10)
    return threshold_expected_connection


def calculate_well_features(graph_representation_per_field, well_data, conn_prob_density, thr_expected_conn):

    short_dst_count, mid_dst_count, long_dst_count, very_long_dst_count, cell_count, valid_fields_count = (0,)*6
    nnl, per_cell_neurite_length_distribution, ercr, dwn, edge_length_distribution = [], [], [], [], []

    for idx, graph_rep_list in enumerate(graph_representation_per_field):
        graph, node_list = graph_rep_list
        num_cells = len(node_list)
        cell_count += num_cells
        valid_fields_count += 1

        nnl.append((well_data["Neurite pixels"][idx] / num_cells))

        field_neu_dst = well_data["Neurite Distribution"][idx]
        per_cell_neurite_length_distribution += field_neu_dst
        field_neu_dst = np.array(field_neu_dst)

        per_cell_num_connections = np.array([d for n, d in graph.degree()])
        per_cell_expected_num_conn = ExpectedNumConnections(node_list, conn_prob_density)
        ercr += list(per_cell_num_connections / (per_cell_expected_num_conn + 0.0001))

        dwn.append(pr_disconnected_with_neurite(per_cell_num_connections, field_neu_dst, per_cell_expected_num_conn, thr_expected_conn))

        field_edge_length_distribution = [weight for (n1, n2, weight) in graph.edges.data("weight")]

        edge_length_distribution += field_edge_length_distribution




        short_dist_pairs_count, mid_dist_pairs_count, long_dist_pairs_count, very_long_dist_pairs_count = denomenator(node_list)
        short_dst_count += short_dist_pairs_count
        mid_dst_count += mid_dist_pairs_count
        long_dst_count += long_dist_pairs_count
        very_long_dst_count += very_long_dist_pairs_count

    short_edges, mid_edges, long_edges, very_long_edges = nomerator(np.array(edge_length_distribution))

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


def calculate_plate_outgrowth_measures(folder_path, conn_prob_density, thr_expected_conn):
    """
    Calculates neurite-outgrowth and toxicity measures
    for each well from the data extracted in the computer vision pipeline.

    Parameters
    ----------
    folder_path: str
        Path to a folder that contains the copmuter vision pipeline output:
        pickle files for each well and one txt file.
    conn_prob_density: ndarray
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
        well_features = calculate_well_features(graph_per_field, well_data, conn_prob_density, thr_expected_conn)
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

    db = DBSCAN(eps=100, min_samples=10)
    # calculate the connection probability density function from negative reference wells data
    connection_pdf = calculate_connection_pdf(folder_path, negative_ref_wells, db)
    # calculate expected connection thresholds
    thr_expected_connection = calculate_thresholds(folder_path, negative_ref_wells, db, connection_pdf)
    # extract neurite outgrowth measures and outlier information from each well in the plate
    plate_processed_data = calculate_plate_outgrowth_measures(folder_path, connection_pdf, thr_expected_connection)
    return plate_processed_data