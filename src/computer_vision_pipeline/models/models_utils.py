import random, tensorflow as tf, numpy as np
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)
from src.computer_vision_pipeline.models.Mask_RCNN.config import Config

### Nucleus Detection Model Utils ###
#####################################

class BowlConfig(Config):
    """
    Configuration for data science bowl model.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"

    IMAGE_RESIZE_MODE = "pad64"  ## tried to modfied but I am using other git clone
    ## No augmentation
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False  ## Not using this

    IMAGE_MIN_DIM = 512  # We scale small images up so that smallest side is 512
    IMAGE_MAX_DIM = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_NMS_THRESHOLD = 0.2
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.001

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nuclei

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600
    PRE_NMS_LIMIT = 6000

    USE_MINI_MASK = True
    DETECTION_MAX_INSTANCES = 1000

### Live Neutire segmentation Model Utils ###
#############################################

from keras import backend as K

ALPHA = 0.5
BETA = 0.5

# setting loss functions (needed to load the model)
def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


def Tverskymetric(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return Tversky






