# Geometry-Aware Spatio-Temporal Context Modeling for 4D Occupancy Forecasting (ECCV 2026)

## Environment

1. Create the conda environment
```setup
conda create -n occ python=3.7
conda activate occ
```

2. Install the following dependencies
```setup
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.1/index.html
pip install spconv-cu116 tensorboard numba einops pyquaternion nuscenes-devkit
```

## Prepare Dataset

1. Download Occ3D-nuScenes from [Occ3D-nuScenes](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link).

2. Download the generated info file from [Google Drive](https://drive.google.com/drive/folders/1nXMyP9FueCuJ-1kFLvFbzvdnaUkwL1L6?usp=drive_link).

## Model Zoo

| Method | Input | Ego traj. | mIoU (%) | IoU (%) |                                                Weights                                                | 
|:------:|:-----:|:---------:|:--------:|:-------:|:-----------------------------------------------------------------------------------------------------:|
|  GAST  | 3D-Occ |  Pred.  |  27.38  |  35.90  | [Google-drive](https://drive.google.com/file/d/1S-aqLxJqTt7Zhe9u_7xpf4QMbCgkcPR1/view?usp=sharing) |
|  GAST  | 3D-Occ |   GT    |  47.40  |  56.24  | [Google-drive](https://drive.google.com/file/d/1cV4c4zyoiitZcJEdvFhr6aiCWwT3mI4-/view?usp=sharing) |

## Training and Evaluation

1. Modify the configuration file `config/gast_nuscenes.yaml`.

2. Enter the `tasks/occ3d_nuscenes` folder, modify `run.sh` and run the command `./run.sh`.

## Acknowledgements

Thanks to the excellent projects: [OccWorld](https://github.com/wzzheng/OccWorld), [II-World](https://github.com/lzzzzzm/II-World)
