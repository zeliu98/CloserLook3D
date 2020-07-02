# Installation

## Preparation

### Requirements
- `Ubuntu 16.04`
- `Anaconda` with `python=3.6`
- `tensorFlow=1.12`
- `cuda=9.0`
- `cudnn>=7.4`
- others: `pip install termcolor opencv-python toposort h5py easydict`

### Compile custom operators
```bash
sh init.sh
```

### Datasets
**Shape Classification on ModelNet40**

You can download ModelNet40 for [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) (1.6 GB). Unzip and move (or link) it to `data/ModelNet40/modelnet40_normal_resampled`.

**Part Segmentation on PartNet**

You can download PartNet dataset from [the ShapeNet official webpage](https://www.shapenet.org/download/parts) (8.0 GB). Unzip and move (or link) it to `data/PartNet/sem_seg_h5`.

**Scene Segmentation on S3DIS**

You can download the S3DIS dataset from [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2) (4.8 GB). You only need to download the file named `Stanford3dDataset_v1.2.zip`, unzip and move (or link) it to `data/S3DIS/Stanford3dDataset_v1.2`.

The file structure should look like:
```
<tf-code-root>
├── cfgs
│   ├── modelnet
│   ├── partnet
│   └── s3dis
├── data
│   ├── ModelNet40
│   │   └── modelnet40_normal_resampled
│   │       ├── modelnet10_shape_names.txt
│   │       ├── modelnet10_test.txt
│   │       ├── modelnet10_train.txt
│   │       ├── modelnet40_shape_names.txt
│   │       ├── modelnet40_test.txt
│   │       ├── modelnet40_train.txt
│   │       ├── airplane
│   │       ├── bathtub
│   │       └── ...
│   ├── PartNet
│   │   └── sem_seg_h5
│   │       ├── Bag-1
│   │       ├── Bed-1
│   │       ├── Bed-2
│   │       ├── Bed-3
│   │       ├── Bottle-1
│   │       ├── Bottle-3
│   │       └── ...
│   └── S3DIS
│       └── Stanford3dDataset_v1.2
│           ├── Area_1
│           ├── Area_2
│           ├── Area_3
│           ├── Area_4
│           ├── Area_5
│           └── Area_6
├── init.sh
├── datasets
├── function
├── models
├── ops
└── utils
```
  

## Usage

### Training

#### ModelNet
```bash
python function/train_evaluate_modelnet.py --cfg <config file>  \
    [--gpus <list of gpus>] [--log_dir <log directory>]
```
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<list of gpus>` means the indexes of gpus you want to use for training, like `0`, `0 1`, `0 1 2 3`.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log`.

#### PartNet
```bash
python function/train_evaluate_partnet.py --cfg <config file>  \
    [--gpus <list of gpus>] [--log_dir <log directory>]
```

#### S3DIS
```bash
python function/train_evaluate_s3dis.py --cfg <config file>  \
    [--gpus <list of gpus>] [--log_dir <log directory>]
```

### Evaluating

#### ModelNet40
```bash
python function/train_evaluate_modelnet.py --cfg <config file> --load_path <checkpoint> \
    [--gpu <gpu>] [--log_dir <log directory>]
```
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<checkpoint>` is the model checkpoint used for evaluating.
- `<gpu>` means which gpu you want to use for evaluating, note that we only use one gpu for evaluating.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log_eval`.

#### PartNet
```bash
python function/evaluate_partnet.py --cfg <config file> --load_path <checkpoint> \
    [--gpu <gpu>] [--log_dir <log directory>]
```

#### S3DIS
```bash
python function/evaluate_s3dis.py --cfg <config file> --load_path <checkpoint> \
    [--gpu <gpu>] [--log_dir <log directory>]
```

# Models

|Method | ModelNet40 | S3DIS | PartNet (val/test)| 
|:---:|:---:|:---:|:---:|
|Point-wise MLP| [92.8](https://drive.google.com/drive/folders/1-CLOVeSmsA-M6sORRhg9mXWRFqloc-n5?usp=sharing) | [66.2](https://drive.google.com/drive/folders/1a02JAQnx3WbZ2ngICNSG6fkZ-Mq_knm7?usp=sharing) | [48.1/51.2](https://drive.google.com/drive/folders/13xAG9D6L0bBBXM8kS6pDYEzApOlae7TZ?usp=sharing) |
|Pseudo Grid| [93.0](https://drive.google.com/drive/folders/16fUdp41jSDD9kHUrXCk_v_1LEkLFsOAp?usp=sharing) | [65.9](https://drive.google.com/drive/folders/1DDkYHliFwlwKkloqBUXke0qP125yXurD?usp=sharing) | [50.8/53.0](https://drive.google.com/drive/folders/1cGdr1vnYB1ZkMjDUfUZ2YCfoscr5NyV0?usp=sharing) |
|Adapt Weights| [93.0](https://drive.google.com/drive/folders/1KSVZvPdqTE0I6fceIx7y6Xp1uaxEd5R3?usp=sharing) | [66.5](https://drive.google.com/drive/folders/1UtSqxg1Bbfk21Rq7SuLllVbm0rDa3Bli?usp=sharing) | [50.1/53.5](https://drive.google.com/drive/folders/1am3oRLnvj5crHXkLl00gAmdE54vGucPe?usp=sharing) |
|PosPool| [92.9](https://drive.google.com/drive/folders/1O34VC4APga7hykrNVuoeL8n0LCLNIv96?usp=sharing) | [66.5](https://drive.google.com/drive/folders/15xzFso1lZy-WrAx57eJ7wQd4zYTYPMfz?usp=sharing) | [50.0/53.4](https://drive.google.com/drive/folders/1KxjArkFtRUCDkVU8CLO3halL1sU3aZ0N?usp=sharing) |
|PosPool*| [93.2](https://drive.google.com/drive/folders/10P_Gu5cmaJqg4VyXa27iDi32LH4rpc7o?usp=sharing) | [66.7](https://drive.google.com/drive/folders/1MT-262K2m65mkUq077DhrFTMqnM-XdEa?usp=sharing) | [50.6/53.8](https://drive.google.com/drive/folders/1eKfuVctpSiAsIpdT0JA3Ns5Su0UhN0QA?usp=sharing) |
