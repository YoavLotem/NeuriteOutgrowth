import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_location(well_name, char2num):
    """
    calculates for a well's name its row and column indices in an array that represents the plate.

    Parameters
    ----------
    well_name: str
        the name of the well in the form "B - 02" when 2 is the column and B is the row
    char2num: dict
        A dictionary indicating the order of the different rows from left to right in the plate

    Returns
    -------
    row, col : int
        indices (starting count from zero) of the well in an array that represents the plate
    """
    row = char2num[well_name[0]]
    col = int(well_name[4:]) - 1
    return row, col

def create_heatmap(arr, feature, mask, max_value):
    """
    Creates an heatmap and display it.

    Parameters
    ----------
    arr: ndarray
     an 8x12 numpy array that represents the well values of a given feature in a plate
    feature: str
        The name of the feature to display its distribution
    mask: ndarray
        ndarray of type boolean. True values where NOT to display results
    max_value: int
        maximal value to show in cbar

    """
    yticks = ["A", "B", "C", "D", "E", "F", "G", "H"]
    xticks = np.arange(1, 13, 1)
    cmap = "Blues"

    fig, ax = plt.subplots(figsize=(10, 10))
    g = sns.heatmap(arr, xticklabels=xticks, yticklabels=yticks, linewidths=1,
    linecolor="black", square=True, cmap=cmap, cbar=True, vmin=0,
    vmax=max_value, annot=True, mask=mask, cbar_kws={"shrink": 0.54,"aspect": 17}, ax=ax)

    g.set_facecolor('white')
    plt.yticks(rotation=0, fontsize="x-large")
    plt.xticks(fontsize="x-large")
    plt.title(feature)


def visualize_plate(plate_processed_data, feature, outlier=True, max_value=1):
    """
    Visualize a feature's distribution across the wells of a plate

    Parameters
    ----------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    feature: str
        The name of the feature to display its distribution
    outlier: bool
        indicates whether the feature is an outlier feature
    max_value: int
        maximal value to show in cbar

    """
    char2num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}
    plate = np.zeros((8, 12))
    mask = np.ones((8, 12), dtype=bool)
    for well_name in plate_processed_data:
        if outlier:
            assert feature in plate_processed_data[well_name]["outlier_dictionary"], "feature is not an outlier feature"
            data_point = plate_processed_data[well_name]["outlier_dictionary"][feature]
        else:
            data_point = -1
            if feature in plate_processed_data[well_name]:
                data_point = plate_processed_data[well_name][feature]

        row, col = get_location(well_name, char2num)
        plate[row, col] = data_point
        mask[row, col] = False
    create_heatmap(arr=plate, feature=feature, mask=mask, max_value=max_value)
    plt.show()



def calculate_z_score(plate_processed_data, feature):
    """
    Calculates z-score for every well in the plate for a given feature

    Parameters
    ----------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    feature: str
        The name of the feature to display its distribution

    Returns
    -------
    z_scores: dict
        A dict containing the well names as keys and their corresponding z-scores as values

    """
    feature_scores = {}
    for well_name in plate_processed_data:
        if feature in plate_processed_data[well_name]:
            feature_scores[well_name] = plate_processed_data[well_name][feature]
        else:
            feature_scores[well_name] = None

    feature_values = list(feature_scores.values())
    feature_mean = np.mean(feature_values)
    feature_std = np.std(feature_values)
    z_scores = {}
    for well_name in feature_scores:
        feature_val = feature_scores[well_name]
        if feature_val:
            z_score = (feature_val-feature_mean)/feature_std
            z_scores[well_name] = z_score
        else:
            z_scores[well_name] = None
    return z_scores

def visualize_hits(plate_processed_data, feature, figsize=(5,5), k=2):
    """
    Calculates z-score for each well in the plate for a given feature and display the results.

    Parameters
    ----------
    plate_processed_data: dict
        dictionary containing each wells neurite outgrowth toxicity and outlier information
    feature: str
        The name of the feature to display its distribution
    figsize: tuple
        tuple (h,w) indicating the width and height of the figure displaying the results
    k: int
        The number of standard deviations required for a hit (could also be a negative number)


    """
    z_scores = calculate_z_score(plate_processed_data, feature)
    plt.figure(figsize=figsize)
    plt.scatter(z_scores.values(), z_scores.keys())
    plt.title(feature + " Z-score per Well")
    plt.vlines(x=k, colors='g', ymin=0, ymax=len(z_scores), linestyles='dashed')
    plt.show()