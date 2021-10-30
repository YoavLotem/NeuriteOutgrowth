import networkx as nx
import cv2
import os
import numpy as np
from skimage import morphology
from skan import Skeleton, summarize
from skimage.morphology import watershed
from src.computer_vision_pipeline.utils import save_pickle, append_dict, sortWells, isSaved, byFieldNumber
from src.computer_vision_pipeline.models.load_models import nucModel, neurite_model
from src.computer_vision_pipeline.segmentation.foreground_segmentation import segment_foreground
from src.computer_vision_pipeline.segmentation.nuclei_instance_segmentation import segment_nuclei
from src.computer_vision_pipeline.segmentation.neurite_semantic_segmentation import segment_neurites
from src.computer_vision_pipeline.segmentation.backscatter import quntifyBackscatter
from src.computer_vision_pipeline.graph.graph_representation import create_graph


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
    graph: Instance of class graph of NetworkX
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
    cells_foreground_mask = segment_foreground(fitc_image)

    # perform nuclei instance segmentation and extract each individual nucleus' pixels location,
    # center location and fraction of dead cells
    nuclei_instance_segmentation_mask, centroids, apoptosis_fraction = segment_nuclei(dapi_image, cells_foreground_mask, nucModel)

    # perform neurite semantic segmentation - which pixels in the FITC image belongs to a neurite
    neurite_mask = segment_neurites(fitc_image, neurite_model)

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
    graph, neurite_length_dict = create_graph(graph, cell_instance_segmentation_mask, skeleton_branch_data, centroids)

    # return all the extracted information
    number_of_cells = len(neurite_length_dict)
    data = {"Cell Number": float(number_of_cells), "Neurite pixels": float(total_neurite_length),
            "Apoptosis Fraction": apoptosis_fraction, "Backscatter": float(backscatter),
            "Neurite Distribution": list(neurite_length_dict.values())}
    return graph, nodes_dict, data


def extract_data_from_plate_images(folder, saving_folder, fields_per_well=20):
    """
    Extracts and saves neurite outgrowth features from all the images in a folder that
    contain images from a single plate.

    Parameters
    ----------
    folder: str
        Path to a folder that contains all the plate's images
    saving_folder: str
        Path specifying where to save the results
    fields_per_well: int, default=20
        Number of fields in each well.

    """
    # create a path to save a txt file that will hold (most of) the extracted plate data
    txt_saving_path = os.path.join(saving_folder, os.path.basename(os.path.normpath(folder)) + '.txt')

    # create a list of the FITC images in the plate each represents a different field of view
    field_names = [s for s in os.listdir(folder) if 'FITC' in s]

    # create a list of the unique well names in the plate, sorted by row and column
    unique_well_names = sortWells(list(set([s[:6] for s in field_names])))

    # initiate a boolean to indicate that no information has yet to be saved in this run
    # allows running multiple times and pick up the run from the last saved well
    first_iter = True

    for well_name in unique_well_names:
        # initiate dictionaries to hold the information of each well
        well_data = {well_name: {"Cell Number": [], "Neurite pixels": [], "Apoptosis Fraction": [], "Backscatter": [], "Neurite Distribution": []}}
        graph_representation_dictionary = {well_name: []}
        # a path to save the graph embeddings as a pickle file
        pkl_saving_path = os.path.join(saving_folder, well_name)

        # in case this isnt the first run (last run was partial), the following conditions allows us to return to
        # the last well that did not finish its calculations (extracted data wasn't saved)

        # if a txt file exists and first_iter==True -> not the first call to function
        if os.path.isfile(txt_saving_path) and first_iter:
            # check whether the current well was already processed and its information was saved in a previous run
            if isSaved(txt_saving_path, well_name):
                continue

        # create a list of the currently processed well fields (FITC images names) and sort them by field number
        fields_of_current_well = [s for s in field_names if well_name in s]
        fields_of_current_well.sort(key=byFieldNumber)

        # make sure the well has the specified number of fields for each well
        assert len(fields_of_current_well) == fields_per_well, 'Number of fields in well ' + well_name + ' is ' + str(len(fields_of_current_well)) + " (should have been " + str(fields_per_well) + ")"

        for fitc_image_name in fields_of_current_well:
            # extract neurite outgrowth information from each field of view
            graph, nodes_list, data = single_field_procedure(folder, fitc_image_name)
            graph_representation_dictionary[well_name].append([graph, nodes_list])
            for key in list(well_data[well_name].keys()):
                well_data[well_name][key].append(data[key])

        # save the well-level information:
        # 1) add the well data to the plate txt file
        # 2) graph representations of the well as pickle files
        append_dict(well_data, txt_saving_path)
        save_pickle(pkl_saving_path, graph_representation_dictionary)
        first_iter = False


