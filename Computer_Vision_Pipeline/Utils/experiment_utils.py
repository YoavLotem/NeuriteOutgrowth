import networkx as nx
import cv2
import os
import numpy as np
from skimage import morphology
from skan import Skeleton, summarize
from skimage.morphology import watershed
from Computer_Vision_Pipeline.Utils.utils import save_pickle, append_dict, sort_wells, continueOrNot, byFieldNum
from Computer_Vision_Pipeline.Utils.Loading_Models import nucModel, neurite_model
from Computer_Vision_Pipeline.Utils.segmentation_utils import segmentNuclei, segmentNeurites, segmentForeground
from Computer_Vision_Pipeline.Utils.image_utils import Backscatter_flag
from Computer_Vision_Pipeline.Utils.graph_representation_utils import create_graph

def single_field_procedure(folder, image_name):
    G = nx.Graph()
    Fitc_image = cv2.imread(os.path.join(folder, image_name), 0)
    dapi_image = cv2.imread(os.path.join(folder, image_name.replace('FITC', 'DAPI')))
    segmented_image = segmentForeground(Fitc_image)
    nuc_im, centroids, apoptosis_fraction = segmentNuclei(dapi_image, segmented_image, nucModel)
    nodes_list = {nuc_num: tuple(centroids[nuc_num])[::-1] for nuc_num in range(1, len(centroids))}
    G.add_nodes_from(nodes_list)
    mask = segmentNeurites(Fitc_image, neurite_model)
    seg = watershed(-1*Fitc_image, nuc_im, watershed_line=True, mask=segmented_image)
    Y, X = np.ogrid[:2048, :2048]
    num_neurites, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    neu = np.zeros((2048, 2048))
    neu[labels >= 1] = 1
    skeleton0 = morphology.skeletonize(neu)
    neurite_sum = np.sum(skeleton0)
    Backscatter = Backscatter_flag(dapi_image)
    if neurite_sum == 0:
        neurite_dist = [0] * len(nodes_list)
        data = {"Cell Number": float(len(nodes_list)), "Neurite pixels": float(neurite_sum),
                "Apoptosis Fraction": apoptosis_fraction, "Backscatter": float(Backscatter),
                "Neurite Distribution": neurite_dist}
        return G, nodes_list, data
    branch_data = summarize(Skeleton(skeleton0))
    G, field_dict = create_graph(G, seg, branch_data, centroids)
    num_nucs = len(field_dict)
    data = {"Cell Number": float(num_nucs), "Neurite pixels": float(neurite_sum),
            "Apoptosis Fraction": apoptosis_fraction, "Backscatter": float(Backscatter),
            "Neurite Distribution": list(field_dict.values())}
    return G, nodes_list, data

def extract_data_from_plate_images(folder, saving_folder, fields_per_well=20):
    saving_path = os.path.join(saving_folder, os.path.basename(os.path.normpath(folder)) + '.txt')
    fNames = [s for s in os.listdir(folder) if 'FITC' in s]
    wNames = sort_wells(list(set([s[:6] for s in fNames])))
    first_iter = True

    for well_name in wNames:
        well_data = {well_name: {"Cell Number": [], "Neurite pixels": [], "Apoptosis Fraction": [], "Backscatter": [],
                                 "Neurite Distribution": []}}
        graph_embedding = {well_name: []}
        pkl_saving_path = os.path.join(saving_folder, well_name)
        if os.path.isfile(saving_path) and first_iter:
            if continueOrNot(saving_path, well_name):
                continue

        fNames_temp = [s for s in fNames if well_name in s]
        fNames_temp.sort(key=byFieldNum)
        assert len(fNames_temp) == fields_per_well, 'number of fields in well is not valid'

        for image_name in fNames_temp:
            field_G, nodes_list, data = single_field_procedure(folder, image_name)
            graph_embedding[well_name].append([field_G, nodes_list])
            for name in list(well_data[well_name].keys()):
                well_data[well_name][name].append(data[name])

            print(image_name, ' ', data["Cell Number"])

        append_dict(well_data, saving_path)
        save_pickle(pkl_saving_path, graph_embedding)
        first_iter = False
