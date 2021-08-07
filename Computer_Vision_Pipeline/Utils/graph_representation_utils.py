from Computer_Vision_Pipeline.common import IMAGE_WIDTH, RADIUS, DISK_MASK, SEARCH_MASK
import numpy as np

def placeSearchDisk(neurite_endpoint, SEARCH_MASK, DISK_MASK):
    """
    Places the DISK_MASK in the right place in the SEARCH_MASK, that is intended to search for cells in the proximity
    of a neurite endpint, so that the center of the search disk will be at the neurite endpoint.
    This approach to set the SEARCH_MASK using the precomputed DISK_MASK is significantly faster than the naive approach

    Parameters
    ----------
    neurite_endpoint: tuple
        (x,y) coordinates of the neurite endpoint
    SEARCH_MASK: ndarray
        2D array containing data with boolean type
        has the same size as the morphology image and searches for cells in the neurite endpoint proximity
    DISK_MASK: ndarray
        2D array containing data with boolean type
        a square mask containing the RADIUS sized search disk

    Returns
    -------
    SEARCH_MASK:ndarray
        2D array containing data with boolean type
        has the same size as the morphology image and searches for cells in the neurite endpoint proximity
        contains a search disk to search for cells surrounding the neurite_endpoint.

    """

    # set up the correct position to place the square mask containing the search disk (if the neurite endpoint is close
    # to one of the image edges the mask will be trimmed because we can't search outside of the image)

    left = max(neurite_endpoint[0] - (RADIUS + 1), 0)
    right = min(neurite_endpoint[0] + (RADIUS + 1), IMAGE_WIDTH - 1)
    down = max(neurite_endpoint[1] - (RADIUS + 1), 0)
    up = min(neurite_endpoint[1] + (RADIUS + 1), IMAGE_WIDTH - 1)

    # calculate how much of the DISK_MASK do we need in case the endpoint is close to the edge
    disc_left = max((RADIUS + 1) - (neurite_endpoint[0] - left), 0)
    disc_right = min(right - neurite_endpoint[0] + (RADIUS + 1), (RADIUS + 1) * 2)
    disc_down = max((RADIUS + 1) - (neurite_endpoint[1] - down), 0)
    disc_up = min(up - neurite_endpoint[1] + (RADIUS + 1), (RADIUS + 1) * 2)

    # place the DISK_MASK with the correct coordinate in the SEARCH_MASK in the correct position
    SEARCH_MASK[down: up, left: right] = DISK_MASK[disc_down: disc_up, disc_left: disc_right]
    return SEARCH_MASK


def calcDistance(centroids, first_idx, second_idx):
    """
    Calculates the euclidean distance between two cells' centers

    Parameters
    ----------
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers
    first_idx:  int
        index of a cell's center coordinates in the centroids array
    second_idx:  int
        index of a cell's center coordinates in the centroids array

    Returns
    -------
    euclidean_distance: float
        The euclidean distance between two cells' centroids
    """
    first_cell_center = np.array(centroids[first_idx])
    second_cell_center = np.array(centroids[second_idx])
    euclidean_distance = np.linalg.norm(first_cell_center - second_cell_center)
    return euclidean_distance


def createEdges(connected_by_neurite, centroids):
    """
    Create a list of edges between cells (nodes) that touch the same neurite (are in the proximity of its endpoints)

    Parameters
    ----------
    connected_by_neurite: set
        A set of unique cell numbers that are in the proximity of a neurite's endpoint
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers

    Returns
    -------
    edges: list
        A list of the form (cell_x, cell_y, {"weight": distance((cell_x, cell_y))}) of all unique pair combinations
        of cells in connected_by_neurite (all cells that are connected via a single neurite)
    """
    connected_by_neurite_list = list(connected_by_neurite)
    num_connected_cells = len(connected_by_neurite_list)
    edges = []
    # iterate other the cells to create an edge between every unique pair of cells that are connected
    for cell_idx in range(num_connected_cells):
        cell_number = connected_by_neurite_list[cell_idx]
        for other_cell_idx in range(cell_idx + 1, num_connected_cells):
            other_cell_number = connected_by_neurite_list[other_cell_idx]
            distance = calcDistance(centroids, cell_number, other_cell_number)
            edges.append((cell_number, other_cell_number, {'weight': int(distance)}))
    return edges

def searchCellsCloseToEndpoint(branch, SEARCH_MASK, DISK_MASK, seg):
    end_point_y = branch.image_coord_dst_0
    end_point_x = branch.image_coord_dst_1
    center = (int(end_point_x), int(end_point_y))
    circle_mask = placeSearchDisk(center, SEARCH_MASK, DISK_MASK)
    cells_close_to_endpoint, counts = np.unique(seg[circle_mask], return_counts=True)
    not_background = cells_close_to_endpoint != 0
    cells_close_to_endpoint = cells_close_to_endpoint[not_background]
    counts = counts[not_background]
    return cells_close_to_endpoint, counts


def findCellWithMaxOverlap(cells_close_to_endpoint, counts):
    max_count = np.argmax(counts)
    closest_cell = cells_close_to_endpoint[max_count]
    return closest_cell


def addSkeletonToGraph(graph, neurite_length_by_skeleton_id, connected_by_neurite, field_dict, skeleton_id, centroids):

    neurite_length = int(neurite_length_by_skeleton_id.loc[skeleton_id])
    if len(connected_by_neurite) > 0:
        normalized_neurite_length = neurite_length / len(connected_by_neurite)
        for cell in connected_by_neurite:
            field_dict[cell] += round(normalized_neurite_length, 2)
        edges = createEdges(connected_by_neurite, centroids)
        graph.add_edges_from(edges)
    return graph, field_dict


def createGraph(graph, seg, skeleton_branch_data, centroids):
    # change name format to allow for itertuples loop and use of column names as attributes
    skeleton_branch_data = skeleton_branch_data.rename(columns={name: name.replace('-', '_') for name in skeleton_branch_data.columns})
    # create a dictionary to hold each cell's neurite length pixels (cell wise neurite length)
    field_dict = {cell_num: 0 for cell_num in range(1, np.max(seg) + 1)}
    # group branches based on skeleton_id
    df_grouped_by_id = skeleton_branch_data.groupby(['skeleton_id'])
    # get the length of each individual neurite (number of pixels of its skeleton)
    neurite_length_by_skeleton_id = df_grouped_by_id['branch_distance'].sum()

    # iterate other all skeletonized connected components
    for skeleton_id, skeleton_data in df_grouped_by_id:
        # initialize a set that will hold unique cells that are in the proximity of the neurite's skeleton endpoints
        connected_by_neurite = set()
        # remove branches which are not endpoints
        skeleton_endpoint_branches = skeleton_data[(skeleton_data['branch_type'] != 2)]

        # iterate other each branch in the skeleton and search for cells in proximity of endpoints
        for branch in skeleton_endpoint_branches.itertuples(index=False):
            cells_close_to_endpoint, counts = searchCellsCloseToEndpoint(branch, SEARCH_MASK, DISK_MASK, seg)
            if len(cells_close_to_endpoint) == 0:
                continue
            # register only one cell per endpoint as connected - the cell with maximal overlap
            closest_cell = findCellWithMaxOverlap(cells_close_to_endpoint, counts)
            connected_by_neurite.add(closest_cell)

        # after all the cells connected via the neurite's skeleton are found
        # we can insert the edges between them into the graph
        graph, field_dict = addSkeletonToGraph(graph, neurite_length_by_skeleton_id, connected_by_neurite, field_dict, skeleton_id, centroids)
    return graph, field_dict

