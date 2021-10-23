import numpy as np
from sklearn.cluster import DBSCAN

# Image Parameters
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 2048
IMAGE_SHAPE = (2048, 2048)


# Set a boolean mask that will contain a round search area to search for cells in the proximity of neurite endpoints
RADIUS = 15  # size in pixels of the search radius around each neurite endpoint to search for a cell
Square_length = (RADIUS + 1) * 2 + 1
circle_y, circle_x = np.ogrid[: Square_length, : Square_length]
DISK_MASK = (circle_x - (RADIUS + 1)) ** 2 + (circle_y - (RADIUS + 1)) ** 2 <= RADIUS ** 2


# Outlier Removal

MIN_VALID_FIELDS = 5

# Outlier removal thresholds
MIN_CELL_NUM = 50  # min allowed number of cells in a field
MAX_CELL_NUM = 1000  # max allowed number of cells in a field
MAX_APOP_RATIO = 0.25  # max allowed ratio of un-viable cells in a field
MAX_HIGH_DENSITY_RATIO = 0.45  # max allowed ratio of extremely clustered cells

# Density Parameters
D_EPS = 100
MIN_SAMPLES = 10

# smoothing constant
EPS = 0

# setting distance constants
MIN_DISTANCE = 0
MAX_DISTANCE = 1000
BIN_SIZE = 25

# RANSAC constants
RANSAC_MIN_SAMPLES = 5
PROBABILITY_THRESHOLD = 0.05

assert RANSAC_MIN_SAMPLES <= MIN_VALID_FIELDS, "The minimal number of valid fields has to be equal or larger" \
                                               " than the number of minimal ransac samples or else the algorithm night" \
                                               " not work"


