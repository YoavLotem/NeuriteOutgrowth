import numpy as np


class ExperimentConfig:

    # smoothing constant
    EPS = 0

    # Image dimensions
    # -------------------------------------------------------------------------
    IMAGE_WIDTH = 2048
    IMAGE_HEIGHT = 2048
    IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

    # Parameters for a boolean mask that containing a round search area to search
    # for cells in the proximity of neurite endpoints
    # ----------------------------------------------------------------------------

    # length in pixels of the search radius around each neurite endpoint to search for a cell
    RADIUS = 15

    # square boolean mask edge length
    square_edge_length = (RADIUS + 1) * 2 + 1
    y, x = np.ogrid[: square_edge_length, : square_edge_length]

    # boolean mask with disk of ones at the center
    DISK_MASK = (x - (RADIUS + 1)) ** 2 + (y - (RADIUS + 1)) ** 2 <= RADIUS ** 2

    # Outlier Removal
    # ----------------------------------------------------------------------------

    # minimum number of fields to accept the results of a well as valid
    MIN_VALID_FIELDS = 5

    # Outlier removal thresholds:
    # minimal number of cells allowed in a field for it to be valid
    MIN_CELL_NUM = 50

    # maximal number of cells allowed in a field for it to be valid
    MAX_CELL_NUM = 1000

    # max allowed ratio of un-viable cells in a field
    MAX_APOP_RATIO = 0.25

    # max allowed ratio of extremely clustered cells
    MAX_HIGH_DENSITY_RATIO = 0.45

    # Parameters for cell density:
    # a cell in a highly dense area in the field is a cell with
    # at least MIN_SAMPLES in a range of D_EPS raduis around it
    D_EPS = 100
    MIN_SAMPLES = 10

    # unsupervised outlier removal constants:
    # straight line will be calculated using Random sample consensus (RANSAC) with
    # number of samples randomly selected equal to RANSAC_MIN_SAMPLES.
    RANSAC_MIN_SAMPLES = 5
    assert RANSAC_MIN_SAMPLES <= MIN_VALID_FIELDS, "The minimal number of valid fields has to be equal or larger" \
                                                   " than the number of minimal ransac samples or else" \
                                                   " the algorithm might not work"
    # fields with residual distance far away will have a low probability to fit the RANSAC line
    # fields with probability lower than threshold will be considered un-valid.
    PROBABILITY_THRESHOLD = 0.05

    # connection probability over a distance (connection_pdf) constants:
    # minimal and maximal distances for calculating the probability of connection
    MIN_DISTANCE = 0
    MAX_DISTANCE = 1000

    # distance range of each pdf bin - meaning the probability of connection will be calculated in
    # the following distance ranges to create the connection_pdf:
    # (MIN_DISTANCE : BIN_SIZE),
    # ((MIN_DISTANCE + BIN_SIZE) : (MIN_DISTANCE + 2*BIN_SIZE)),
    # ...
    # (MAX_DISTANCE - BIN_SIZE) : MAX_DISTANCE) range
    BIN_SIZE = 25

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")




