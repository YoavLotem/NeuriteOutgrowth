import networkx as nx, pickle, os, numpy as np, ast, scipy, random, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from scipy.stats import norm
from itertools import compress
import seaborn as sns

def well_names_generator(folder):
    well_names = [file_name for file_name in os.listdir(folder) if '.txt' not in file_name]
    healthy_well_names = [well_name for well_name in well_names if well_name[0] in ["A", "B", "C", "D"]]
    FD_well_names = [well_name for well_name in well_names if well_name[0] in ["E", "F", "G", "H"]]
    random.shuffle(healthy_well_names)
    random.shuffle(FD_well_names)
    return healthy_well_names, FD_well_names

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
    ransac = RANSACRegressor(residual_threshold=1 * std, min_samples=5, random_state=42)
    ransac.fit(X, y)
    yhat = ransac.predict(X)
    R = y - yhat
    mu, std = norm.fit(R)
    n = norm(mu, std)
    inliers = n.cdf(R) > 0.05
    return inliers[:, 0]


def ExpectedNumConnections(node_list, prob):
    node_arr = np.array(list(node_list.values()))
    d = scipy.spatial.distance.pdist(node_arr)
    d = scipy.spatial.distance.squareform(d)
    expected_num_conn = np.zeros(len(node_arr))
    distances_list = np.arange(50, 1050, 50)
    last_distance = 0
    for distance in distances_list:
        neighbours_in_distance = np.sum(np.logical_and(d < distance, d >= last_distance), axis=1)
        expected_num_conn += neighbours_in_distance * prob[(distance - 50) // 50]  # calculating the index from the distance
        last_distance = distance
    return expected_num_conn


def ExpectedConnectionsLength(node_list, prob):
    node_arr = np.array(list(node_list.values()))
    d = scipy.spatial.distance.pdist(node_arr)
    d = scipy.spatial.distance.squareform(d)
    expected_num_conn = np.zeros(len(node_arr))
    distances_list = np.arange(25, 1050, 25)
    last_distance = 0
    for distance in distances_list:
        neighbours_in_distance = np.sum(np.logical_and(d < distance, d >= last_distance), axis=1)
        expected_num_conn += neighbours_in_distance * prob[(distance - 25) // 25] * distance  # calculating the index from the distance
        last_distance = distance
    return expected_num_conn


def connectivity_per_distance_pdf(node_list, edge_lenghts):
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


def pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn, threshold_expected_connection):
    pr_disconnected_with_neu_not_isolated = np.sum(np.logical_and(np.logical_and(degrees == 0, temp_image_neu_dst != 0), expected_num_conn > threshold_expected_connection))
    pr_with_neu_not_isolated = np.sum(np.logical_and(temp_image_neu_dst != 0, expected_num_conn > threshold_expected_connection))
    conditional = pr_disconnected_with_neu_not_isolated / (pr_with_neu_not_isolated + 0.00001)
    return (len(degrees) ** 0.5) * conditional
    # return conditional


def pr_disco_without_neu(degrees, temp_image_neu_dst, expected_num_conn, threshold_expected_connection_high):
    pr_disconnected_with_neu_not_clustered = np.sum(np.logical_and(np.logical_and(degrees == 0, temp_image_neu_dst == 0), expected_num_conn < threshold_expected_connection_high))
    pr_with_neu_not_clustered = np.sum(np.logical_and(temp_image_neu_dst == 0, expected_num_conn < threshold_expected_connection_high))
    conditional = pr_disconnected_with_neu_not_clustered / (pr_with_neu_not_clustered + 0.00001)
    return conditional / (len(degrees) ** 0.5)


def unpack_dictionary(line, well_name):
    d = ast.literal_eval(line)
    assert list(d.keys())[0] == well_name, 'well name not compatable'
    num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = d[well_name]["Cell Number"], d[well_name]["Neurite pixels"], d[well_name]["Apoptosis Fraction"], d[well_name]["Backscatter"], \
                                                                          d[well_name]["Neurite Distribution"]
    return num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist


def distance_metric(node_list, distance):
    node_arr = np.array(list(node_list.values()))
    d = scipy.spatial.distance.pdist(node_arr)
    d = scipy.spatial.distance.squareform(d)
    neighbours_mask = d < distance
    neighbours = np.sum(neighbours_mask, axis=1)
    return neighbours


def load_pickle(full_path):
    with open(full_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content


def denomenator(node_list):
    node_arr = np.array(list(node_list.values()))
    d1 = scipy.spatial.distance.pdist(node_arr)
    d = scipy.spatial.distance.squareform(d1)
    short_dist = np.sum(d1 <= 100)
    mid_dist = np.sum(np.logical_and(d1 > 100, d1 <= 300))
    long_dist = np.sum(np.logical_and(d1 > 300, d1 <= 400))
    very_long = np.sum(d1 > 300)
    return short_dist, mid_dist, long_dist, very_long


def nomerator(edge_lenghts):
    short_edges = np.sum(edge_lenghts <= 100)
    mid_edges = np.sum(np.logical_and(edge_lenghts > 100, edge_lenghts <= 300))
    long_edges = np.sum(np.logical_and(edge_lenghts > 300, edge_lenghts <= 400))
    very_long_edges = np.sum(edge_lenghts > 300)
    return short_edges, mid_edges, long_edges, very_long_edges


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


def get_reference_connection_probability_density(folder, ref_names, db):
    pkl_files = [os.path.join(folder, file_name) for file_name in os.listdir(folder) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder) if '.txt' in name][0]
    lines = [line.rstrip('\n') for line in open(os.path.join(folder, txt_file_name))]
    prob_ref = []
    for line, file_full_path in zip(lines, pkl_files):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        if well_name not in ref_names:
            continue
        list_of_graph_embedings = well_dictionary[well_name]
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)

        result_tupple, _ = total_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)
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


def extract_features_specific_references(folder, h_prob, s_prob, threshold_expected_connection_h, threshold_expected_connection_high_h, threshold_expected_connection_s, threshold_expected_connection_high_s, db):
    pkl_files = [os.path.join(folder, file_name) for file_name in os.listdir(folder) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder) if '.txt' in name][0]
    lines = [line.rstrip('\n') for line in open(os.path.join(folder, txt_file_name))]
    exp_data = {}
    for line, file_full_path in zip(lines, pkl_files):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)
        list_of_graph_embedings = well_dictionary[well_name]
        short, mid, long1, very_long, disconnected_with_neu_h, disconnected_with_neu_s, disconnected_without_neu_h, disconnected_without_neu_s, edge_lenghts = 0, 0, 0, 0, [], [], [], [], []
        cell_counter, valid_fields = 0, 0
        neu2cell_ratio, neu_dist_undiscarded, expectedVSreal_num_conn_h, expectedVSreal_num_conn_s = [], [], [], []
        result_tupple, outlier_dict = total_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)

        if len(result_tupple) == 1:
            exp_data[well_name] = {"outlier_dictionary": outlier_dict}
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = result_tupple

        for idx, graph_and_node_list in enumerate(list_of_graph_embedings):

            graph = graph_and_node_list[0]
            node_list = graph_and_node_list[1]
            num_cells = len(node_list)
            cell_counter += num_cells
            valid_fields += 1

            temp_image_neu_dst = neu_dist[idx]
            neu_dist_undiscarded += temp_image_neu_dst
            temp_image_neu_dst = np.array(temp_image_neu_dst)
            degrees = np.array([d for n, d in graph.degree()])

            neu2cell_ratio.append((neu_pix_list[idx] / num_cells))

            assert len(degrees) == len(temp_image_neu_dst)

            expected_num_conn_h = ExpectedNumConnections(node_list, h_prob)
            expectedVSreal_num_conn_h += list(degrees / (expected_num_conn_h + 0.0001))
            expected_num_conn_s = ExpectedNumConnections(node_list, s_prob)
            expectedVSreal_num_conn_s += list(degrees / (expected_num_conn_s + 0.0001))
            disconnected_with_neu_h.append(pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn_h, threshold_expected_connection_h))
            disconnected_with_neu_s.append(pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn_s, threshold_expected_connection_s))

            disconnected_without_neu_h.append(pr_disco_without_neu(degrees, temp_image_neu_dst, expected_num_conn_h, threshold_expected_connection_high_h))
            disconnected_without_neu_s.append(pr_disco_without_neu(degrees, temp_image_neu_dst, expected_num_conn_s, threshold_expected_connection_high_s))

            length_list = [weight for (n1, n2, weight) in graph.edges.data("weight")]

            edge_lenghts += length_list
            short_dist, mid_dist, long_dist, very_long_dist = denomenator(node_list)
            short += short_dist
            mid += mid_dist
            long1 += long_dist
            very_long += very_long_dist

        short_edges, mid_edges, long_edges, very_long_edges = nomerator(np.array(edge_lenghts))
        exp_data[well_name] = {"expectedVSreal_num_conn_h": np.mean(expectedVSreal_num_conn_h), "expectedVSreal_num_conn_s": np.mean(expectedVSreal_num_conn_s), "Valid Fields": valid_fields,
                               "neu2cells_ratio": np.mean(np.array(neu2cell_ratio)), "Neurite Average": np.mean(neu_dist_undiscarded),
                               "Disconnected Without Neurites healthy ref": np.mean(disconnected_without_neu_h),"Disconnected Without Neurites fd ref": np.mean(disconnected_without_neu_s), "Disconnected With Neurites healthy ref": np.nanmean(disconnected_with_neu_h),
                               "Disconnected With Neurites fd ref": np.nanmean(disconnected_with_neu_s), "Short Connection Probability": short_edges / short,
                               "Intermediate Connection Probability": mid_edges / mid, "Long Connection Probability": long_edges / long1,
                               "Very Long Connection Probability": very_long_edges / very_long, "# Cells": cell_counter, "outlier_dictionary": outlier_dict}
    return exp_data


def visualize_histograms(exp_data):
    names = list(exp_data["D - 05"].keys())
    names.remove("outlier_dictionary")
    valid_wells = []
    for well_name in exp_data:
        if "Valid Fields" in exp_data[well_name]:
            if exp_data[well_name]["Valid Fields"] >= 5:
                valid_wells.append(well_name)

    # names = ["Neurite Average", "Disconnected With Neurites",	"Short Connection Probability",	"Intermediate Connection Probability",	"Long Connection Probability",	"Very Long Connection Probability",	"# Cells"]
    mat = np.zeros((len(valid_wells), len(names)))
    for col, name in enumerate(names):
        h = []
        s = []
        for row, well_name in enumerate(valid_wells):
            prob = exp_data[well_name][name]
            mat[row, col] = prob
            if well_name[0] in ['A', 'B', 'C', 'D']:
                h.append(prob)
            else:
                s.append(prob)
        ratio = round(np.mean(h) / np.mean(s), 2)
        # if name == "Disconnected Without Neurites":
        plt.figure()
        plt.hist(h, alpha=0.5, label="Healthy")
        plt.hist(s, alpha=0.5, label="Sick")
        plt.title("Histogram of " + name + " (mean ratio = " + str(ratio) + ")")
        plt.legend()
        plt.savefig(name + '.png', dpi=1000, bbox_inches='tight')
        plt.show()


def create_outlier_heatmap(arr, outlier_type, mask, exp_name=None, cbar=False, y_label_white=False):
    yticks = ["A", "B", "C", "D", "E", "F", "G", "H"]
    purples = {"Number of Fields With under 50 Cells", "Number of Fields With over 1000 Cells", "Number of Fields With Clustered Cells", "Number of RANSAC Outlier Fields", "Number of Fields With Apoptosis"}
    xticks = np.arange(1, 13, 1)
    vmax = 20 if "Apoptosis Ratio" not in outlier_type else 1
    cmap = 'Greys' if "Apoptosis Ratio" in outlier_type else "YlOrBr" if outlier_type in purples else "Blues"

    # plt.figure(figsize=(8,8))
    # if cbar:
    #     yticks = []
    # sns.set_theme()

    g = sns.heatmap(arr, xticklabels=xticks, yticklabels=yticks, linewidths=1, linecolor="black", square=True, cmap=cmap, cbar=False, vmin=0, vmax=vmax, annot=True, mask=mask, cbar_kws={"shrink": 0.54,"aspect": 17})
    g.set_facecolor('white')
    plt.yticks(rotation=0, fontsize="x-large")
    if y_label_white:
        plt.tick_params(axis='y', colors='white')
    plt.xticks(fontsize="x-large")
    plt.title(exp_name)
    # plt.savefig(os.path.join(r'C:\Users\t-yolote\OneDrive - Microsoft\Documents\school\thesis\save images\outlier removal plate ' + exp_name[-3], outlier_type + '.png'), dpi=1000, bbox_inches='tight')
    # plt.show()

def get_location(well_name, char2num):
    row = char2num[well_name[0]]
    col = int(well_name[4:]) - 1
    return (row, col)

def visualize_outliers_as_plate(exp_data_temp, exp_name=None, cbar=False):
    char2num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
    plate = np.zeros((8, 12, 8))
    mask = np.ones((8, 12), dtype=bool)
    mask[1:7, 1:11] = False
    keys = ['Number of Fields With under 50 Cells', 'Number of Fields With over 1000 Cells', 'Number of Fields With Apoptosis', 'Apoptosis Ratio Before Outlier Removal', 'Number of Fields With Clustered Cells', 'Number of RANSAC Outlier Fields', 'Apoptosis Ratio After Outlier Removal', 'Valid Fields']
    for well_name in exp_data_temp:
        outlier_dict = exp_data_temp[well_name]["outlier_dictionary"]
        if "Valid Fields" not in outlier_dict:
            outlier_dict["Valid Fields"] = exp_data_temp[well_name]["Valid Fields"]
        location = get_location(well_name, char2num)
        plate[location[0], location[1], :] = np.array([outlier_dict[key] for key in keys])
    for idx, key in enumerate(keys):
        create_outlier_heatmap(arr=plate[:, :, idx], outlier_type=key, mask=mask, exp_name=exp_name, cbar=cbar)
        plt.show()

def extract_features_specific_references_per_fields(folder, h_prob, s_prob, threshold_expected_connection_h, threshold_expected_connection_high_h, threshold_expected_connection_s, threshold_expected_connection_high_s, db):
    pkl_files = [os.path.join(folder, file_name) for file_name in os.listdir(folder) if '.txt' not in file_name]
    txt_file_name = [name for name in os.listdir(folder) if '.txt' in name][0]
    lines = [line.rstrip('\n') for line in open(os.path.join(folder, txt_file_name))]
    exp_data = {}
    for line, file_full_path in zip(lines, pkl_files):
        well_dictionary = load_pickle(file_full_path)
        well_name = list(well_dictionary.keys())[0]
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist = unpack_dictionary(line, well_name)
        list_of_graph_embedings = well_dictionary[well_name]
        short, mid, long1, very_long, disconnected_with_neu_h, disconnected_with_neu_s, disconnected_without_neu_h, disconnected_without_neu_s, edge_lenghts = [], [], [], [], [], [], [], [], []
        cell_counter, valid_fields = 0, 0
        neu2cell_ratio, neu_dist_undiscarded, expectedVSreal_num_conn_h, expectedVSreal_num_conn_s = [], [], [], []
        short_edges_list, mid_edges_list, long_edges_list, very_long_edges_list = [], [], [], []
        result_tupple, outlier_dict = total_outlier_removal(num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings, db, well_name)
        cell_counter = []
        if len(result_tupple) == 1:
            exp_data[well_name] = {"outlier_dictionary": outlier_dict}
            continue
        num_cells_list, neu_pix_list, apop_list, backscatter_list, neu_dist, list_of_graph_embedings = result_tupple

        for idx, graph_and_node_list in enumerate(list_of_graph_embedings):

            graph = graph_and_node_list[0]
            node_list = graph_and_node_list[1]
            num_cells = len(node_list)
            cell_counter.append(num_cells)
            valid_fields += 1

            temp_image_neu_dst = neu_dist[idx]
            neu_dist_undiscarded.append(np.mean(temp_image_neu_dst))
            temp_image_neu_dst = np.array(temp_image_neu_dst)
            degrees = np.array([d for n, d in graph.degree()])

            neu2cell_ratio.append((neu_pix_list[idx] / num_cells))

            assert len(degrees) == len(temp_image_neu_dst)

            expected_num_conn_h = ExpectedNumConnections(node_list, h_prob)
            expectedVSreal_num_conn_h.append(np.mean(list(degrees / (expected_num_conn_h + 0.0001))))
            expected_num_conn_s = ExpectedNumConnections(node_list, s_prob)
            expectedVSreal_num_conn_s.append(np.mean(list(degrees / (expected_num_conn_s + 0.0001))))
            disconnected_with_neu_h.append(pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn_h, threshold_expected_connection_h))
            disconnected_with_neu_s.append(pr_disconnected_with_neurite(degrees, temp_image_neu_dst, expected_num_conn_s, threshold_expected_connection_s))

            disconnected_without_neu_h.append(pr_disco_without_neu(degrees, temp_image_neu_dst, expected_num_conn_h, threshold_expected_connection_high_h))
            disconnected_without_neu_s.append(pr_disco_without_neu(degrees, temp_image_neu_dst, expected_num_conn_s, threshold_expected_connection_high_s))

            length_list = [weight for (n1, n2, weight) in graph.edges.data("weight")]

            short_edges, mid_edges, long_edges, very_long_edges = nomerator(np.array(length_list))

            short_edges_list.append(short_edges)
            mid_edges_list.append(mid_edges)
            long_edges_list.append(long_edges)
            very_long_edges_list.append(very_long_edges)




            short_dist, mid_dist, long_dist, very_long_dist = denomenator(node_list)
            short.append(short_dist)
            mid.append(mid_dist)
            long1.append(long_dist)
            very_long.append(very_long_dist)


        exp_data[well_name] = {"expectedVSreal_num_conn_h": expectedVSreal_num_conn_h, "expectedVSreal_num_conn_s": expectedVSreal_num_conn_s, "Valid Fields": valid_fields,
                               "neu2cells_ratio": neu2cell_ratio, "Neurite Average": neu_dist_undiscarded,
                               "Disconnected Without Neurites healthy ref": disconnected_without_neu_h,"Disconnected Without Neurites fd ref": disconnected_without_neu_s, "Disconnected With Neurites healthy ref": disconnected_with_neu_h,
                               "Disconnected With Neurites fd ref": disconnected_with_neu_s, "Short Connection Probability": np.array(short_edges_list) / np.array(short),
                               "Intermediate Connection Probability": np.array(mid_edges_list) / np.array(mid), "Long Connection Probability": np.array(long_edges_list) / np.array(long1),
                               "Very Long Connection Probability": np.array(very_long_edges_list) / np.array(very_long), "# Cells": cell_counter, "outlier_dictionary": outlier_dict}
    return exp_data