# depth_matching

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

Download the dataset:

```
cd Dataset

./raw_data_downloader.sh
```

Install dependencies:

```
pip install matplotlib pykitti pillow opencv-python numpy open3d
```

To install pytorch:

```
pip3 install torch torchvision torchaudio
```