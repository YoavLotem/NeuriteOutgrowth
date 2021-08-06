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


def nuc_dist(centroids, first_idx, second_idx):
    a = np.array(centroids[first_idx])
    b = np.array(centroids[second_idx])
    return np.linalg.norm(a - b)


def create_edges(touching_cells, centroids):
    touching_cells_list = list(touching_cells)
    L = len(touching_cells_list)
    edges = []
    for cell_idx in range(L):
        nuc_num_cell = touching_cells_list[cell_idx]
        for other_cell_idx in range(cell_idx + 1, L):
            nuc_num_other_cell = touching_cells_list[other_cell_idx]
            distance = nuc_dist(centroids, nuc_num_cell, nuc_num_other_cell)
            edges.append((nuc_num_cell, nuc_num_other_cell, {'weight': int(distance)}))
    return edges


def create_graph(G, seg, branch_data, centroids):
    branch_data = branch_data.rename(columns={name: name.replace('-', '_') for name in branch_data.columns})
    branch_data = branch_data.sort_values(by=['skeleton_id'])
    field_dict = {cell_num: 0 for cell_num in range(1, np.max(seg) + 1)}


    df_grouped_by_id = branch_data.groupby(['skeleton_id'])
    neurite_length_by_skeleton_id = df_grouped_by_id['branch_distance'].sum()
    skeleton_endpoint_branches = branch_data[(branch_data['branch_type'] != 2)]

    touching_cells = set()
    last_id = False
    for branch in skeleton_endpoint_branches.itertuples(index=False):
        current_id = branch.skeleton_id
        if last_id:
            if current_id != last_id:
                L = len(touching_cells)
                neurite_length = int(neurite_length_by_skeleton_id.loc[last_id])
                if L > 0:
                    normalized_neurite_length = neurite_length / len(touching_cells)
                    for cell in touching_cells:
                        field_dict[cell] += round(normalized_neurite_length, 2)
                    edges = create_edges(touching_cells, centroids)
                    G.add_edges_from(edges)
                touching_cells = set()
        end_point_y = branch.image_coord_dst_0
        end_point_x = branch.image_coord_dst_1
        center = (int(end_point_x), int(end_point_y))
        circle_mask = placeSearchDisk(center, SEARCH_MASK, DISK_MASK)
        cells_close_to_endpoint, counts = np.unique(seg[circle_mask], return_counts=True)
        not_background = cells_close_to_endpoint != 0
        if np.sum(not_background) == 0:
            continue
        cells_close_to_endpoint = cells_close_to_endpoint[not_background]
        counts = counts[not_background]
        max_count = np.argmax(counts)
        closest_cell = cells_close_to_endpoint[max_count]
        touching_cells.add(closest_cell)
        last_id = branch.skeleton_id
    return G, field_dict