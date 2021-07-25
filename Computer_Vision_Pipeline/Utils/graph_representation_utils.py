import numpy as np


def create_circular_mask(center, mask, tiny_mask, radius):
    radiusplusone = radius + 1
    left = max(center[0] - radiusplusone, 0)
    right = min(center[0] + radiusplusone, 2047)

    down = max(center[1] - radiusplusone, 0)
    up = min(center[1] + radiusplusone, 2047)

    tiny_left = max(radiusplusone - (center[0] - left), 0)
    tiny_right = min(right - center[0] + radiusplusone, radiusplusone * 2)
    tiny_down = max(radiusplusone - (center[1] - down), 0)
    tiny_up = min(up - center[1] + radiusplusone, radiusplusone * 2)
    mask[down: up, left: right] = tiny_mask[tiny_down: tiny_up, tiny_left: tiny_right]
    return mask, down, up, left, right


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
    unique_skeletons = branch_data['skeleton_id'].unique()
    field_dict = {cell_num: 0 for cell_num in range(1, np.max(seg) + 1)}

    circle_mask = np.full((2048, 2048), False)
    radius = 15
    rectangle_edge_length = (radius + 1) * 2 + 1
    circle_y, circle_x = np.ogrid[: rectangle_edge_length, : rectangle_edge_length]
    tiny_mask = (circle_x - (radius + 1)) ** 2 + (circle_y - (radius + 1)) ** 2 <= radius ** 2

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
        circle_mask, down, up, left, right = create_circular_mask(center, circle_mask, tiny_mask, radius)
        cells_close_to_endpoint, counts = np.unique(seg[circle_mask], return_counts=True)
        circle_mask[down: up, left: right] = False
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