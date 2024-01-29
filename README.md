<div align="center">

<h2>Towards Generating a Probabilistic Ensemble of 3D Rigid Body Simulations from an RGBD Image Using Gaussian Processes</h2>

<div>
  Onur Beker<sup>1</sup>&emsp;
  Yunhan Wang<sup>1</sup>&emsp;
  Barbu Bojor<sup>1</sup>&emsp;
</div>

<div>
    <sup>1</sup>University of Tübingen
</div>


</div>

## Description
This repository contains the accompanying code for our final project report for the course Data Literacy (ML4201, 6 ECTS / 4 SWS) at the University of Tübingen.

## Setting up the Environment
The main dependencies can be installed by running the following command in a python virtual environment:
```
pip install torch torchvision open3d opencv-python scipy pillow tqdm GPy PyMCubes matplotlib plotly
```
Please note that as we use the [DINOv2](https://github.com/facebookresearch/dinov2) backbone to obtain feature descriptors from RGBD images, running the code requires a cuda-enabled GPU with minimum 3.5 GB VRAM.

## Getting started 
- The file `app/dataset_filtering_experiments.ipynb` contains the code for the preprocessing, dataset creation, and data processing steps of our pipeline. It produces the same visualizations that were used in Fig.1 and Fig.2 of our project report.
- The file `app/dataset_filtering_experiments.ipynb` contains the code for fitting a Gaussian process (GP) to the created dataset and sampling from its posterior to build the probabilistic ensemble of likely 3D reconstructions. It produces the same visualizations that were used in Fig.3 of our project report.
 





