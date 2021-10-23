from common import IMAGE_WIDTH, RADIUS, DISK_MASK
import numpy as np

def place_search_disk(neurite_endpoint, search_mask, DISK_MASK):
    """
    Places the DISK_MASK in the right place in the search_mask, that is intended to search for cells in the proximity
    of a neurite endpoint, so that the center of the search disk will be at the neurite endpoint.
    This approach to set the SEARCH_MASK using the precomputed DISK_MASK is significantly faster than the naive approach

    Parameters
    ----------
    neurite_endpoint: tuple
        (x,y) coordinates of the neurite endpoint
    search_mask: ndarray
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
    up, down, right, left: int
        pixel coordinates where the DISK_MASK was placed
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
    search_mask[down: up, left: right] = DISK_MASK[disc_down: disc_up, disc_left: disc_right]
    return search_mask, down, up, left, right


def calculate_distance(centroids, first_idx, second_idx):
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


def create_edges(connected_by_neurite, centroids):
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
            distance = calculate_distance(centroids, cell_number, other_cell_number)
            edges.append((cell_number, other_cell_number, {'weight': int(distance)}))
    return edges


def search_cells_close_to_endpoint(branch, search_mask, DISK_MASK, soma_inst_seg_mask):
    """
    Identify cells in the proximity of a neurite endpoint

    Parameters
    ----------
    branch: namedtuple
        A namedtuple that contains information of a single branch which is part
        of a skeletonized connected component of a neurite
    search_mask: ndarray
        2D array containing data with boolean type
        has the same size as the morphology image and searches for cells in the neurite endpoint proximity
    DISK_MASK: ndarray
        2D array containing data with boolean type
        a square mask containing the RADIUS sized search disk
    soma_inst_seg_mask: ndarray
        2D array containing data with int type
        Cell-body instance segmentation mask. Each individual cell has a different integer
        assigned to the pixels it appears at. Background value is zero.

    Returns
    -------
    cells_close_to_endpoint: ndarray
        1D array containing data with int type
        contains the identifying numbers of cells that are in the proximity of the neurite endpoint
    counts: ndarray
        1D array containing data with int type
        Contains the amount of pixels that that are in the proximity of the endpoint
        for each of the cells mentioned in cells_close_to_endpoint.
        This data helps finding out which cell is connected to the neurite via the current endpoint.

    """
    # the x,y coordinates of the endpoint of the branch (not the source of the branch)
    end_point_y = branch.image_coord_dst_0
    end_point_x = branch.image_coord_dst_1
    center = (int(end_point_x), int(end_point_y))
    # place the search disk in the right place in the search mask
    # in order to look for cells in the proximity of the endpoint in a fast way
    search_mask, down, up, left, right = place_search_disk(center, search_mask, DISK_MASK)
    # look for unique pixel values in the proximity of the endpoint
    # (each different pixel value in the cell body instance segmentation mask represents a different cell)
    cells_close_to_endpoint, counts = np.unique(soma_inst_seg_mask[search_mask], return_counts=True)
    # retrieve search_mask to its original condition (before placeSearchDisk)
    search_mask[down: up, left: right] = False
    # keep only the values of cells(remove background values)
    not_background = cells_close_to_endpoint != 0
    cells_close_to_endpoint = cells_close_to_endpoint[not_background]
    counts = counts[not_background]
    return cells_close_to_endpoint, counts


def find_cell_with_max_overlap(cells_close_to_endpoint, counts):
    """
    Find the cell with maximal overlap with search disk in the proximity of an endpoint.

    Parameters
    ----------
    cells_close_to_endpoint: ndarray
        1D array containing data with int type
        contains the identifying numbers of cells that are in the proximity of the neurite endpoint
    counts: ndarray
        1D array containing data with int type
        Contains the amount of pixels that that are in the proximity of the endpoint
        for each of the cells mentioned in cells_close_to_endpoint.
        This data helps finding out which cell is connected to the neurite via the current endpoint.

    Returns
    -------
    closest_cell: The number of the cell with the maximal overlap with the maximal overlap with search disk in the proximity of an endpoint.
    """
    max_count = np.argmax(counts)
    closest_cell = cells_close_to_endpoint[max_count]
    return closest_cell


def add_skeleton_to_graph(graph, neurite_length_by_skeleton_id, connected_by_neurite, neurite_length_dict, skeleton_id, centroids):
    """
    Add the edges of cells that are connected via a specific neurite.
    In addition, the function divides the neurite's length between the cells that are connected by it
    as an approximation of each cell's neurite outgrowth.

    Parameters
    ----------
    graph: Instance of class graph of NetworkX
        The graph representation of the cell culture
    neurite_length_by_skeleton_id: Pandas Series
        Holds the total length of a each individual skeletonized neurite connected component
    connected_by_neurite: set
        A set containing the numbers of cells that are connected via the neurite
    neurite_length_dict: dict
        Contains the cell wise neurite length for each cell
    skeleton_id: int
        The identifying number of each neurite's skeleton
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers

    Returns
    -------
    graph: Instance of class graph of NetworkX
        The graph representation of the cell culture after
         updating it with the information from the current neurite's skeleton
    neurite_length_dict: dict
        Contains the cell wise neurite length for each cell after
         updating it with the information from the current neurite's skeleton
    """
    neurite_length = int(neurite_length_by_skeleton_id.loc[skeleton_id])
    # update the graph and cell wise neurite length only if cells are connected to the neurite
    if len(connected_by_neurite) > 0:
        neurite_length_per_cell = neurite_length / len(connected_by_neurite)
        # update each cell's neurite length count
        for cell in connected_by_neurite:
            neurite_length_dict[cell] += round(neurite_length_per_cell, 2)
        # create edges between each pair of cells that is connected to the neurite
        edges = create_edges(connected_by_neurite, centroids)
        graph.add_edges_from(edges)
    return graph, neurite_length_dict


def create_graph(graph, soma_inst_seg_mask, skeleton_branch_data, centroids):
    """
    Build a graph representation of the cell culture in the field of view of the DAPI & Morphology images.

    Parameters
    ----------
    graph: Instance of class graph of NetworkX
        The graph representation of the cell culture in the field.
    soma_inst_seg_mask: ndarray
        2D array containing data with int type
        Cell-body instance segmentation mask. Each individual cell has a different integer
        assigned to the pixels it appears at. Background value is zero.
    skeleton_branch_data: Pandas DataFrame
        A DataFrame Summarising the information of the skeletons of the neurites
    centroids: ndarray
        array containing data with int type
        containing [y,x] coordinates of nuclei centers

    Returns
    -------
    graph: Instance of class graph of NetworkX
        The graph representation of the cell culture
    neurite_length_dict: dict
        Contains the cell wise neurite length for each cell
    """
    # change name format to allow for itertuples loop and use of column names as attributes
    skeleton_branch_data = skeleton_branch_data.rename(columns={name: name.replace('-', '_') for name in skeleton_branch_data.columns})
    # create a dictionary to hold each cell's neurites length in pixels (cell wise neurite length)
    neurite_length_dict = {cell_num: 0 for cell_num in range(1, np.max(soma_inst_seg_mask) + 1)}
    # group branches based on skeleton_id
    df_grouped_by_id = skeleton_branch_data.groupby(['skeleton_id'])
    # get the length of each individual neurite (number of pixels of its skeleton)
    neurite_length_by_skeleton_id = df_grouped_by_id['branch_distance'].sum()
    # initialize a mask to search for cells in the proximity of neurite endpoints
    search_mask = np.full((2048, 2048), False)

    # iterate other all skeletonized connected components
    for skeleton_id, skeleton_data in df_grouped_by_id:
        # initialize a set that will hold unique cells that are in the proximity of the neurite's skeleton endpoints
        connected_by_neurite = set()
        # remove branches which are not endpoints
        skeleton_endpoint_branches = skeleton_data[(skeleton_data['branch_type'] != 2)]

        # iterate other each branch in the skeleton and search for cells in proximity of endpoints
        for branch in skeleton_endpoint_branches.itertuples(index=False):
            cells_close_to_endpoint, counts = search_cells_close_to_endpoint(branch, search_mask, DISK_MASK, soma_inst_seg_mask)
            if len(cells_close_to_endpoint) == 0:
                continue
            # register only one cell per endpoint as connected - the cell with maximal overlap
            closest_cell = find_cell_with_max_overlap(cells_close_to_endpoint, counts)
            connected_by_neurite.add(closest_cell)

        # after all the cells connected via the neurite's skeleton are found
        # we can insert the edges between them into the graph
        graph, neurite_length_dict = add_skeleton_to_graph(graph, neurite_length_by_skeleton_id, connected_by_neurite, neurite_length_dict, skeleton_id, centroids)
    return graph, neurite_length_dict
