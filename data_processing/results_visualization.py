import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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