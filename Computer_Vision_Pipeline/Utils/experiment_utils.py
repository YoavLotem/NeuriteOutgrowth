import networkx as nx
import cv2
import os
import numpy as np
from skimage import morphology
from skan import Skeleton, summarize
from skimage.morphology import watershed
from Computer_Vision_Pipeline.Utils.utils import save_pickle, append_dict, sortWells, isSaved, byFieldNumber
from Computer_Vision_Pipeline.Utils.loading_models import nucModel, neurite_model
from Computer_Vision_Pipeline.Utils.segmentation_utils import segmentNuclei, segmentNeurites, segmentForeground
from Computer_Vision_Pipeline.Utils.image_utils import quntifyBackscatter
from Computer_Vision_Pipeline.Utils.graph_representation_utils import createGraph


def single_field_procedure(folder, fitc_image_name):
    """
    Perform all the procedures required in order to extract neurite outgrowth features from a single field of view
    (that includes a DAPI & FITC images).

    Parameters
    ----------
    folder: str
        path to the folder that contains the field of view images
    fitc_image_name: str
        image name of the form: "B - 02(fld 01 wv FITC - FITC).tif'"

    Returns
    -------
    graph: Instance of class Graph of NetworkX
        The graph representation of the cell culture in the field
    nodes_dict: dict
        A dictionary that contain the field's nuclei centroids X,Y coordinates (which are the graph's nodes)
    data: dict
        A dictionary containing most of the data regarding neurite outgrowth information extracted from the of the field

    """
    # Load the DAPI & FITC images
    fitc_image = cv2.imread(os.path.join(folder, fitc_image_name), 0)
    dapi_image = cv2.imread(os.path.join(folder, fitc_image_name.replace('FITC', 'DAPI')))

    # perform foreground segmentation - which pixels in the FITC image belongs to a cell
    cells_foreground_mask = segmentForeground(fitc_image)

    # perform nuclei instance segmentation and extract each individual nucleus' pixels location,
    # center location and fraction of dead cells
    nuclei_instance_segmentation_mask, centroids, apoptosis_fraction = segmentNuclei(dapi_image, cells_foreground_mask, nucModel)

    # perform neurite semantic segmentation - which pixels in the FITC image belongs to a neurite
    neurite_mask = segmentNeurites(fitc_image, neurite_model)

    # apply the watershed algorithm to achieve cell instance segmentation
    cell_instance_segmentation_mask = watershed(-1*fitc_image, markers=nuclei_instance_segmentation_mask, watershed_line=True, mask=cells_foreground_mask)

    # skeletonize the(binary)neurite mask
    skeletonized_neurite_mask = morphology.skeletonize(neurite_mask)

    # sum the nuerite pixels in the skeletonized neurite mask as an estimation of the total neurite length in the image
    total_neurite_length = np.sum(skeletonized_neurite_mask)

    # check for the backscatter artifact in the DAPI image
    backscatter = quntifyBackscatter(dapi_image)

    # initiate a networkx graph and initiate it with the nuclei centroids as nodes with their (x,y) coordinates
    graph = nx.Graph()
    nodes_dict = {nuc_num: tuple(centroids[nuc_num])[::-1] for nuc_num in range(1, len(centroids))}
    graph.add_nodes_from(nodes_dict)

    # in case there are no neurites in the image we return the information that doesn't depend on the neurites
    if total_neurite_length == 0:
        per_cell_neurite_length_distribution = [0] * len(nodes_dict)
        data = {"Cell Number": float(len(nodes_dict)), "Neurite pixels": float(total_neurite_length),
                "Apoptosis Fraction": apoptosis_fraction, "Backscatter": float(backscatter),
                "Neurite Distribution": per_cell_neurite_length_distribution}
        return graph, nodes_dict, data

    # extract branch-wise information of the connected components of the skeletonized neurite mask
    skeleton_branch_data = summarize(Skeleton(skeletonized_neurite_mask))

    # create a graph representation of the cells and extract a cell wise neurite length dictionary
    graph, neurite_length_dict = createGraph(graph, cell_instance_segmentation_mask, skeleton_branch_data, centroids)

    # return all the extracted information
    number_of_cells = len(neurite_length_dict)
    data = {"Cell Number": float(number_of_cells), "Neurite pixels": float(total_neurite_length),
            "Apoptosis Fraction": apoptosis_fraction, "Backscatter": float(backscatter),
            "Neurite Distribution": list(neurite_length_dict.values())}
    return graph, nodes_dict, data


def extract_data_from_plate_images(folder, saving_folder, fields_per_well=20):
    saving_path = os.path.join(saving_folder, os.path.basename(os.path.normpath(folder)) + '.txt')
    fNames = [s for s in os.listdir(folder) if 'FITC' in s]
    wNames = sortWells(list(set([s[:6] for s in fNames])))
    first_iter = True

    for well_name in wNames:
        well_data = {well_name: {"Cell Number": [], "Neurite pixels": [], "Apoptosis Fraction": [], "Backscatter": [],
                                 "Neurite Distribution": []}}
        graph_embedding = {well_name: []}
        pkl_saving_path = os.path.join(saving_folder, well_name)
        if os.path.isfile(saving_path) and first_iter:
            if isSaved(saving_path, well_name):
                continue

        fNames_temp = [s for s in fNames if well_name in s]
        fNames_temp.sort(key=byFieldNumber)
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
