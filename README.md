# depth_matching

## Setup

Clone the repository with the submodule:
```
git clone https://github.com/cm090999/depth_matching.git --recursive
```

## Conda Setup
First, clone this repository. Then clone the following repositorie into this repository:
``` 
git clone https://github.com/cm090999/LoFTR.git
```

Initialize the conda environment:

``` 
conda env create -f setup/environment.yaml
```

Activate the created environment:
``` 
conda activate depth_matching
```

Download the weights for the LoFTR model from https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf and remove the top level folder to only have a ```weights``` folder in the LoFTR project directory. The ```.chpt``` files should be in the following location:

```
depth_matching/LoFTR/weights/example.ckpt
```

## Get the dataset
Run the bash script to download a subset of the KITTI dataset:

```
cd Dataset
```
```
./raw_data_downloader.sh
```