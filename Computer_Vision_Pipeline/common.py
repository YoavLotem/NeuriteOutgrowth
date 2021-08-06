import numpy as np

# Parameters
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048
IMAGE_SHAPE = (2048, 2048)
RADIUS = 15  # size in pixels of the search radius around each neurite endpoint to search for a cell


# set a boolean mask that will contain a round search area to search for cells in the proximity of neurite endpoints
Square_length = (RADIUS + 1) * 2 + 1
circle_y, circle_x = np.ogrid[: Square_length, : Square_length]
DISK_MASK = (circle_x - (RADIUS + 1)) ** 2 + (circle_y - (RADIUS + 1)) ** 2 <= RADIUS ** 2

# A larger boolean mask that will contatin the SEARCH_DISK_MASK to search the full sizes image
SEARCH_MASK = np.full(IMAGE_SHAPE, False)