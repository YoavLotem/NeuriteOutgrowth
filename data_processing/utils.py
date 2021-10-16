import os
import pickle

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
    """
    load the pickle file containing a per field graph representation of the cell cultures in the current processed well
    and extract its content.

    Parameters
    ----------
    pickle_file_full_path: str
        path to pickle file

    Returns
    -------
    graph_per_field: list
        A list of lists containing for each field in the well a list containing an instance of class graph of NetworkX
        and a dictionary containing the graph's nodes information
    well_name: str
        well name of the form "X - YY" when X is an uppercase character and YY is an integer e.g. "B - 06"

    """
    well_graph_representation_dict = load_pickle(pickle_file_full_path)
    well_name = list(well_graph_representation_dict.keys())[0]
    graph_per_field = well_graph_representation_dict[well_name]
    return graph_per_field, well_name


def load_pickle(full_path):
    with open(full_path, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content