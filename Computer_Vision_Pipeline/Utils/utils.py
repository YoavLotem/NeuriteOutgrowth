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

# The following 3 functions are used as keys to sort through images


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


def get_last_well_name(path):
  with open(path, "r") as file:
    lastline = (list(file)[-1])
  last_dict = ast.literal_eval(lastline)
  last_well_name = list(last_dict.keys())[0]
  return last_well_name

def continueOrNot(path, current_well_name):
  last_well_name = get_last_well_name(path)
  last_char, current_well_name_char = last_well_name[0], current_well_name[0]
  last_number, current_well_name_number = int(last_well_name[4:6]), int(current_well_name[4:6])
  if last_char > current_well_name_char:
    return True
  elif last_char == current_well_name_char and last_number >= current_well_name_number:
    return True
  else:
    return False

def append_dict(dictinary, file_path):
    with open(file_path, 'a') as f:
        json.dump(dictinary, f)
        f.write(os.linesep)
        f.close()

def save_pickle(filename, data):
    file = open(filename, 'wb')
    # dump information to that file
    pickle.dump(data, file)
    # close the file
    file.close()
