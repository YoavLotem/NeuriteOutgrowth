import pickle
import os
import json
import ast

# sorting of images assumes the following structure:
# B - 02(fld 01 wv FITC - FITC).tif
# when:
# B is a the character that represents the row in which the well is in (see plate illustration)
# 02 is the number that represents the the columns in which the well is in (see plate illustration)
# 01 is the field of view (multi channel image) number
######################################################
#       Plate illustration
#   01 02 03 04 05 06 07 08 09 10 11 12
# A  *  *  *  *  *  *  *  *  *  *  *  *
# B  *  *  *  *  *  *  *  *  *  *  *  *
# C  *  *  *  *  *  *  *  *  *  *  *  *
# D  *  *  *  *  *  *  *  *  *  *  *  *
##########################################################


# The following 3 functions are used as keys to sort through wells/fields/images names


def byFieldNumber(elem):
    return int(elem[11:13])


def byRowCharacter(elem):
    return elem[0]


def byColumnNumber(elem):
    return int(elem[4:])


def sortWells(well_names):
    """
    Sorts a list of well names of the form: "B - 02"
    according to their: 1) row character and 2) column number
    for example: sortWells([C - 02, B - 02, B - 05]) -> [B - 02, B - 05, C - 02]

    Parameters
    ----------
    well_names: list
        list of well names of the form: "B - 02"

    Returns
    -------
    sorted_well_names: list
        A sorted list of well names

    """
    # get a list of unique row characters and sort them by characters (A<B)
    row_characters = list(set([name[0] for name in well_names]))
    row_characters.sort(key=byRowCharacter)
    sorted_well_names = []
    # iterate other the sorted unique row characters
    for row_char in row_characters:
        # find every well name with the row character and sort them by column number (1<2)
        same_row_wells = [name for name in well_names if row_char in name]
        same_row_wells.sort(key=byColumnNumber)
        for well in same_row_wells:
            # insert the well names by the order of 1) row character and 2) column number
            sorted_well_names.append(well)
    return sorted_well_names


def getLastSavedWellName(path2json):
    """
    Iterates over the lines of the json file - each line is the data of another well, and returns the name of last well
    that was saved.

    Parameters
    ----------
    path2json: str
        Path to json file that is used to save the experiment's data

    Returns
    -------
    last_well_name: str
        The name of the last well that was saved (well name of the form: "B - 02")
    """
    # extract the last line of the json file which is the data of the last well as a dictionary
    with open(path2json, "r") as file:
        lastline = (list(file)[-1])
    last_dict = ast.literal_eval(lastline)
    # extract the name from the dictionary
    last_well_name = list(last_dict.keys())[0]
    return last_well_name


def isSaved(path2json, current_well_name):
    """
    Allows doing the processing of an experiment in multiple runs by checking whether the current well name
    was already saved to the json file. This is achieved by comparing the current well name to the
    last well name that was saved and relying on the fact that the wells (data) are saved using a specific
    order described in sortWells.

    Parameters
    ----------
    path2json: str
        Path to json file that is used to save the experiment's data
    current_well_name: str
        well name of the form: "B - 02"

    Returns
    -------
    Boolean - True if the current well's data was already saved False if it wasn't
    """
    # extract the name of well that its data was saved last
    last_well_name = getLastSavedWellName(path2json)
    # get the row character and column number of the current and last wells
    last_well_row_char, current_well_row_char = last_well_name[0], current_well_name[0]
    last_well_column_number, current_well_column_number = int(last_well_name[4:6]), int(current_well_name[4:6])
    # compare the two names based on the sorting logic in sortWells
    if last_well_row_char > current_well_row_char:
        return True
    elif last_well_row_char == current_well_row_char and last_well_column_number >= current_well_column_number:
        return True
    else:
        return False


def append_dict(well_data_dictionary, path2json):
    """
    Adds well data dictionary to the end of the json file

    Parameters
    ----------
    well_data_dictionary: dict
        dictionary containing the data of the well
    path2json: str
        Path to json file that is used to save the experiment's data

    """
    with open(path2json, 'a') as f:
        json.dump(well_data_dictionary, f)
        f.write(os.linesep)
        f.close()

def save_pickle(pkl_saving_path, data):
    """
    saves data as a pickle file

    Parameters
    ----------
    pkl_saving_path: str
        path to save the data
    data: object
        any data type - pickle can handle almost everything
    """
    file = open(pkl_saving_path, 'wb')
    # dump information to that file
    pickle.dump(data, file)
    # close the file
    file.close()
