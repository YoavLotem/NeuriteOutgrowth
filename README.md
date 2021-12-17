
#  Neurite Outgrowth Estimation Using Deep Learning & Graph Theory

A tool for neurite outgrowth and toxicity estimation from microscopy images using deep learning and novel graph-based connectivity measures.


![image](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/assets/readme_im_compressed.png)


## Features

- Estimate neurite outgrowth and toxicity for large scale experiments from microscopy images
- Outlier removal algorithms for cleaner results  
- Graph representation of cell cultures and novel connectivity based features for neurite outgrowth
- Neurite semantic segmentation
- Nuclei instance segmentation from [https://github.com/Lopezurrutia/DSB_2018](https://github.com/Lopezurrutia/DSB_2018)
- Cell instance segmentation
- Cell foreground segmentation


## Getting Started

- [Experiment_Demo.ipynb](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/Experiment_Demo.ipynb) Is the easiest way to start analyzing experiment data (e.g. high throuput screening data from a 96 wells plate). 

- [Computer_Vision_Pipeline_Demo.ipynb](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/Computer_Vision_Pipeline_Demo.ipynb) Displays the computer vision models in this repository and the different steps in the neurite outgrowth analysis. 

- (optional) To fully understand the computer vision pipeline, feature extraction, graph representation, novel connectivity features and outlier removal algorithms please refer to the Methods section in [Thesis](https://docs.google.com/document/d/1lT-KUPgt1lQyyrHHMAMJNzhnqgL5ts-7/edit?usp=sharing&ouid=103117274956717598825&rtpof=true&sd=true) or inspect [outlier_removal.py](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/src/data_processing/outlier_removal.py), [feature_extraction.py](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/src/data_processing/feature_extraction.py), [graph_representation.py](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/src/computer_vision_pipeline/graph/graph_representation.py), [experiment_inference_utils.py](https://github.com/YoavLotem/Automatic-Neurite-Outgrowth-Quantification-Using-Deep-Learning/blob/master/src/computer_vision_pipeline/experiment_inference_utils.py). 

## Installation

- Download the [Mask RCNN weights](https://drive.google.com/file/d/1sX5u0dEBvA8Y8z8UObXsty-CE_TjWNKH/view?usp=sharing) (too large for Github) and place them in repository root directory. 
    
## Citation
To DO
