import pickle
import os
import numpy as np
import ast
from data_processing.outlier_removal import total_outlier_removal
from data_processing.feature_extraction import ExpectedNumConnections, pr_disconnected_with_neurite, denomenator, nomerator
from common import EPS

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

def get_thresholds_expected_connection(folder, ref_names, db, prob):
    pkl_files = [os.path.join(folder, file_name) for file_name in os.listdir(folder) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder) if '.txt' in name][0]
    lines = [line.rstrip('\n') for line in open(os.path.join(folder, txt_file_name))]
    expected_conn = []
    for line, file_full_path in zip(lines, pkl_files):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        if well_name not in ref_names:
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)
        list_of_graph_embedings = well_dictionary[well_name]

        result_tupple, _ = total_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)
        if len(result_tupple) == 1:
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = result_tupple

        for idx, graph_and_node_list in enumerate(list_of_graph_embedings):
            node_list = graph_and_node_list[1]
            expected_conn += list(ExpectedNumConnections(node_list, prob))

    threshold_expected_connection = np.percentile(expected_conn, 10)
    threshold_expected_connection_high = np.percentile(expected_conn, 90)
    return threshold_expected_connection, threshold_expected_connection_high


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


def extract_features_specific_references(folder, conn_prob_density, thr_expected_conn):
    pkl_files = [os.path.join(folder, file_name) for file_name in os.listdir(folder) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder) if '.txt' in name][0]
    plate_data_txt = [line.rstrip('\n') for line in open(os.path.join(folder, txt_file_name))]
    plate_features = {}
    for well_data_as_txt, file_full_path in zip(plate_data_txt, pkl_files):
        well_graph_representation_dict = load_pickle(file_full_path)
        well_name = list(well_graph_representation_dict.keys())[0]
        graph_representation_per_field = well_graph_representation_dict[well_name]
        well_data = ast.literal_eval(well_data_as_txt)[well_name]
        result_tuple, outlier_dict = total_outlier_removal(well_data, graph_representation_per_field)
        # check that there are enough valid fields
        if len(result_tuple) == 1:
            plate_features[well_name] = {"outlier_dictionary": outlier_dict}
            continue
        well_data, graph_representation_per_field = result_tuple
        well_features = calculate_well_features(graph_representation_per_field, well_data, conn_prob_density, thr_expected_conn)
        well_features["outlier_dictionary"] = outlier_dict
        plate_features[well_name] = well_features
    return plate_features