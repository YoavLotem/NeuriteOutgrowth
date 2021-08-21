from Computer_Vision_Pipeline.Models.models_utils import *
import Computer_Vision_Pipeline.Models.Mask_RCNN.model as modellib
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
### Nucleus Detection Model ###
###############################
inference_config = BowlConfig()

# Setting up the pre-trained model

model_path = 'deepretina_final.h5'
MODEL_DIR = 'logs'

print("Loading weights from ", model_path)

# Recreate the model in inference mode
nucModel = modellib.MaskRCNN(mode="inference",
                             config=inference_config,
                             model_dir=MODEL_DIR)
nucModel.load_weights(model_path, by_name=True)



### Live Neutire Segmentation Model ###
#######################################
# loading the model (the loss functions are needed only for loading the model)
neurite_model = load_model(r'Old_Architechture_neurite_net_cosine.h5',
                           custom_objects={'TverskyLoss': TverskyLoss, 'Tverskymetric': Tverskymetric})

# change the first layers of the model so that it could accept any size input (it was trained on patches)
neurite_model.layers.pop(0)
newInput = Input((None, None, 1))
newOutputs = neurite_model(newInput)
neurite_model = Model(newInput, newOutputs)
data_gen_args = dict(rescale=1. / 255, samplewise_center=True, data_format="channels_last")