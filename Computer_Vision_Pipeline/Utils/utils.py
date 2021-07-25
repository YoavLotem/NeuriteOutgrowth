import pickle
import os
import json
import ast

def byFieldNum(elem):
    return int(elem[11:13])

def byLetter(elem):
    return elem[0]

def byNum(elem):
    return int(elem[4:])

def sort_wells(wNames):
    L = len(wNames)
    letters = list(set([s[0] for s in wNames]))
    letters.sort(key=byLetter)
    wNames_new = []
    for let in letters:
        let_wells = [s for s in wNames if let in s]
        let_wells.sort(key=byNum)
        for well in let_wells:
            wNames_new.append(well)
    return wNames_new

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
