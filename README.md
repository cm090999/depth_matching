# depth_matching

## Get the dataset
Run the bash script to download a subset of the KITTI dataset:

```
cd Dataset
```
```
./raw_data_downloader.sh
```

## Setup

Clone the repository with the submodule:
```
git clone https://github.com/cm090999/depth_matching.git --recursive
```

Create a virtual environment with '''python 3.10'''

```
python3.10 -m venv .venv
```

Note: If an error occurs, ensure you have installed ```venv```:

```
sudo apt install python3.10-venv
```

Activate the virtual environment:

```
source .venv/bin/activate
```

Install dependencies:

```
pip install matplotlib pykitti pillow opencv-python numpy open3d
```

To install pytorch:

```
pip3 install torch torchvision torchaudio
```

## Conda Setup
Clone the LoFTR repository:
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


