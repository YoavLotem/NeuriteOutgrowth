from src.computer_vision_pipeline.models.models_utils import *
import src.computer_vision_pipeline.models.Mask_RCNN.model as modellib
from src.computer_vision_pipeline.models.Mask_RCNN import utils
from keras.models import load_model
from keras.models import Model
from keras.layers import Input


class CVModels:
    """
    A class used to represent the computer vision models used in the data processing.

    Attributes:
    -----------
    mask_rcnn_weights_path: str
        path to Mask RCNN model weights (h5 file)
    neurite_segmentation_model_path: str
        path to neurite segmentation model (h5 file)

    """
    def __init__(self, mask_rcnn_weights_path, neurite_segmentation_model_path):
        self.mask_rcnn_weights_path = mask_rcnn_weights_path
        self.neurite_segmentation_model_path = neurite_segmentation_model_path

        ### Nucleus Mask RCNN Instance segmentation Model ###
        #####################################################
        inference_config = BowlConfig()

        # Setting up the pre-trained model
        MODEL_DIR = 'logs'

        # Recreate the model in inference mode
        self.nuclei_maskrcnn = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
        self.nuclei_maskrcnn.load_weights(self.mask_rcnn_weights_path, by_name=True)

        ### Live Neutire segmentation Model ###
        #######################################

        # loading the model (the loss functions are needed only for loading the model)
        self.neurite_seg_model = load_model(self.neurite_segmentation_model_path,
                                 custom_objects={'TverskyLoss': TverskyLoss, 'Tverskymetric': Tverskymetric})

        # change the first layers of the model so that it could accept any size input (it was trained on patches)
        self.neurite_seg_model.layers.pop(0)
        newInput = Input((None, None, 1))
        newOutputs = self.neurite_seg_model(newInput)
        self.neurite_seg_model = Model(newInput, newOutputs)
