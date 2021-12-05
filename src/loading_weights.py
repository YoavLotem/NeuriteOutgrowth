import urllib.request
import shutil


mask_rcnn_weights = "https://drive.google.com/file/d/1sX5u0dEBvA8Y8z8UObXsty-CE_TjWNKH/view?usp=sharing"

def download_trained_weights(local_weights_path, verbose=1):
    """
    Download deep retina trained weights.

    Parameters
    ----------
    local_weights_path: str
        local path of trained weights
    verbose: bool
        bool indicating whether to write messages
    """
    if verbose > 0:
        print("Downloading pretrained model to " + local_weights_path + " ...")
    with urllib.request.urlopen(mask_rcnn_weights) as resp, open(local_weights_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")
