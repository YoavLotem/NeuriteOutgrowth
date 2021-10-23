import numpy as np
import scipy
from common import MIN_DISTANCE, MAX_DISTANCE, BIN_SIZE


def calculate_expected_number_of_connections(node_dict, connection_pdf):
    """
    Calculate the expected number of connections for each cell in a field given the cell's location and the connection
    probability density function.

    Parameters
    ----------
    node_dict: dict
        A dictionary that contain the field's nuclei centroids X,Y coordinates (which are the graph's nodes)
    connection_pdf: ndarray
        1d ndarray containing the discrete connection probability density function for each distance
        (0-1000 pixels in 25 pixels bins)

    Returns
    -------
    per_cell_expected_num_connections: ndarray
        A 1D array of the expected number of connections for every cell in a field
    """

    node_arr = np.array(list(node_dict.values()))
    # getting pairwise cell distances in square form - each row holds the distances from one cell to all other cells
    pairwise_distances_condensed = scipy.spatial.distance.pdist(node_arr)
    pairwise_distances_squareform = scipy.spatial.distance.squareform(pairwise_distances_condensed)

    # initializing an array to hold the expected number of connections per cell
    per_cell_expected_num_connections = np.zeros(len(node_arr))

    # initializing an array with the discrete distances of the connection PDF
    distances_arr = np.arange(MIN_DISTANCE, MAX_DISTANCE, BIN_SIZE)

    # iterating over the distances and indices of bins in the connection pdf
    for index_of_bin_in_pdf, distance in enumerate(distances_arr):

        # checking for each cell (axis=1) how many cells are within a distance range from it.
        neighbours_in_distance_range = np.sum(np.logical_and(pairwise_distances_squareform < (distance + BIN_SIZE),
                                                       pairwise_distances_squareform >= distance), axis=1)

        # if the distance range is 0-25 than we need to deduct the cell itself from the count of cells in its proximity
        if distance == 0:
            neighbours_in_distance_range -= 1

        # connection probability in the current distance range is the value of the connection pdf in the current index
        probability_of_connection_in_distance_range = connection_pdf[index_of_bin_in_pdf]

        # multiplying how many neighbors exists with what is the connection probability to a single cell
        # within a the current distance range, contributes to the overall expected number of connections
        per_cell_expected_num_connections += neighbours_in_distance_range * probability_of_connection_in_distance_range
    return per_cell_expected_num_connections


def count_pairs(node_dict):
    """
    Counts (for a field) how many pairs of cells are in each of 4 specific distance ranges [pixels]
    (0-100, 100-300, 300-400, 300-inf).

    Parameters
    ----------
    node_dict: dict
        A dictionary that contain the field's nuclei centroids X,Y coordinates (which are the graph's nodes)

    Returns
    -------
    num_short_distance_pairs: int
        number of cell pairs in a 0-100 pixels distance
    num_medium_distance_pairs: int
        number of cell pairs in a 100-300 pixels distance
    num_long_distance_pairs: int
        number of cell pairs in a 300-400 pixels distance
    num_very_long_distance_pairs: int
        number of cell pairs in a 300 - inf pixels distance
    """
    node_arr = np.array(list(node_dict.values()))
    pairwise_distances_condensed = scipy.spatial.distance.pdist(node_arr)
    num_short_distance_pairs = np.sum(pairwise_distances_condensed <= 100)

    num_medium_distance_pairs = np.sum(np.logical_and(pairwise_distances_condensed > 100,
                                                      pairwise_distances_condensed <= 300))

    num_long_distance_pairs = np.sum(np.logical_and(pairwise_distances_condensed > 300,
                                                    pairwise_distances_condensed <= 400))

    num_very_long_distance_pairs = np.sum(pairwise_distances_condensed > 300)
    return num_short_distance_pairs, num_medium_distance_pairs, num_long_distance_pairs, num_very_long_distance_pairs


def count_connections(edge_lengths):
    """
    Counts the number of connections in a well in each of 4 different distance ranges [pixels]
    (0-100, 100-300, 300-400, 300-inf).

    Parameters
    ----------
    edge_lengths: ndarray
       ndarray containing the length distribution of connections via neurites in a well

    Returns
    -------
    short_edges_count: int
        number of connected cell pairs in a 0-100 pixels distance
    medium_edges_count: int
        number of connected cell pairs in a 100-300 pixels distance
    long_edges_count: int
        number of connected cell pairs in a 300-400 pixels distance
    very_long_edges_count: int
                number of connected cell pairs in a 300-inf pixels distance
    """
    short_edges_count = np.sum(edge_lengths <= 100)
    medium_edges_count = np.sum(np.logical_and(edge_lengths > 100, edge_lengths <= 300))
    long_edges_count = np.sum(np.logical_and(edge_lengths > 300, edge_lengths <= 400))
    very_long_edges_count = np.sum(edge_lengths > 300)
    return short_edges_count, medium_edges_count, long_edges_count, very_long_edges_count


def calculate_disconnected_with_neurites(num_connections, neurite_distribution, expected_num_conn, thr_expected_conn):
    """
    Calculates the probability of a cell (in a field) to be disconnected (not connected via a neurite) given
    that it has a neurite and that it is not isolated (its expected number of connections is higher than threshold) -
    P(disconnected|(has_neurites & not_isolated)).
    Parameters
    ----------
    num_connections: ndarray
        A 1D array of the number of connection of every cell in the field
    neurite_distribution: ndarray
        A 1D array of the number of neurite pixels associated with each cell in the field
    expected_num_conn: ndarray
        A 1D array of the expected number of connections for every cell in the field
    thr_expected_conn: float
        threshold for expected connections - cells with values lower than this are considered "isolated" in the
        calculation of "expected vs real connection ratio" feature.

    Returns
    -------

    """
    # create boolean arrays of the different conditions
    has_neurites = neurite_distribution != 0
    not_isolated = expected_num_conn > thr_expected_conn
    disconnected = num_connections == 0

    # boolean arrays for combinations of conditions
    has_neurites_not_isolated = np.sum(np.logical_and(has_neurites, not_isolated))
    disconnected_and_not_isolated = np.logical_and(disconnected, not_isolated)

    # calculate the number of cells in the field which don't have a connection AND have a neurite AND are not isolated
    disconnected_not_isolated_and_with_neu = np.sum(np.logical_and(disconnected_and_not_isolated, has_neurites))

    # the conditional probability of P(disconnected|(has_neurites & not_isolated)) can be calculated as:
    # P(disconnected|(has_neurites & not_isolated)) = P(disconnected & (has_neurites & not_isolated)) / P(has_neurites & not_isolated)
    # when P(conditions) = (cells that fit conditions)/(number of cells in field)
    # therefore the number of cells in field cancel out

    conditional_probability = disconnected_not_isolated_and_with_neu / (has_neurites_not_isolated + 0.00001)

    # normalizing the feature with a square root of the number of cells in the field
    # (less disconnected cells in dense cultures)
    dwn = (len(num_connections) ** 0.5) * conditional_probability
    return dwn




def calculate_connection_pdf_for_a_single_field(node_dict, edge_lengths):
    """
    Calulculates the probability of connection for a single field over multiple of distance ranges
    (25 [pixels] bins covering the 0-1000 [pixels] range)

    Parameters
    ----------
    node_dict: dict
        A dictionary that contain the field's nuclei centroids X,Y coordinates (which are the graph's nodes)
    edge_lengths: ndarray
       ndarray containing the length distribution of connections via neurites in a field

    Returns
    -------
    connection_pdf_field: ndarray
         1d ndarray containing the discrete connection probability density function for each distance range
        (0-1000 pixels in 25 pixels bins) for a single field
    """
    node_arr = np.array(list(node_dict.values()))
    pairwise_distances_condensed = scipy.spatial.distance.pdist(node_arr)
    # bins for the connection PDF
    distances_arr = np.arange(MIN_DISTANCE, MAX_DISTANCE, BIN_SIZE)
    # array to keep probability values from the current field
    connection_pdf_field = np.zeros(len(distances_arr))

    for idx, distance in enumerate(distances_arr):
        # count how many cell pairs exists within the distance range of distance - distance + BIN_SIZE
        cell_pairs_in_distance_range = np.sum(np.logical_and(pairwise_distances_condensed >= distance,
                                     pairwise_distances_condensed < (distance + BIN_SIZE)))

        # count how many connected cell pairs exists within the distance range of distance - distance + BIN_SIZE
        connected_cell_pairs_in_distance_range = np.sum(np.logical_and(edge_lengths >= distance,
                                                                       edge_lengths < (distance + BIN_SIZE)))


        if cell_pairs_in_distance_range == 0:
            # if there are no cell pairs in this distance range than the probability of connection is 0
            connection_pdf_field[idx] = 0
        else:
            # we define the probability of connection (within the current distance range) as the
            # ratio between the amount of connected cells to the overall number of cell pairs
            connection_pdf_field[idx] = connected_cell_pairs_in_distance_range / cell_pairs_in_distance_range
    return connection_pdf_field

