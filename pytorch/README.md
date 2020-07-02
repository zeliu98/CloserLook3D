# Installation

## Preparation

### Requirements
- `Ubuntu 16.04`
- `Anaconda` with `python=3.6`
- `pytorch>=1.3`
- `torchvision` with  `pillow<7`
- `cuda=10.1`
- others: `pip install termcolor opencv-python tensorboard h5py easydict`
- note


### Datasets
**Shape Classification on ModelNet40**

You can download ModelNet40 for [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) (1.6 GB). Unzip and move (or link) it to `data/ModelNet40/modelnet40_normal_resampled`.

**Part Segmentation on PartNet**

You can download PartNet dataset from [the ShapeNet official webpage](https://www.shapenet.org/download/parts) (8.0 GB). Unzip and move (or link) it to `data/PartNet/sem_seg_h5`.

**Scene Segmentation on S3DIS**

You can download the S3DIS dataset from [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2") (4.8 GB). You only need to download the file named `Stanford3dDataset_v1.2.zip`, unzip and move (or link) it to `data/S3DIS/Stanford3dDataset_v1.2`.

The file structure should look like:
```
<pt-code-root>
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

### Compile custom operators and pre-processing data
```bash
sh init.sh
```

## Usage

### Training

#### ModelNet
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    function/train_modelnet_dist.py --cfg <config file> [--log_dir <log directory>]
```
- `<port_num>` is the port number used for distributed training, you can choose like 12347.
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log`.

#### PartNet
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    function/train_partnet_dist.py --cfg <config file> [--log_dir <log directory>]
```

#### S3DIS
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    function/train_s3dis_dist.py --cfg <config file> [--log_dir <log directory>]
```

### Evaluating
For evaluation, we recommend using 1 gpu for more precise result.
#### ModelNet40
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    function/evaluate_modelnet_dist.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
 ```
- `<port_num>` is the port number used for distributed evaluation, you can choose like 12347.
- `<config file>` is the yaml file that determines most experiment settings. Most config file are in the `cfgs` directory.
- `<checkpoint>` is the model checkpoint used for evaluating.
- `<log directory>` is the directory that the log file, checkpoints will be saved, default is `log_eval`.

#### PartNet
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    function/evaluate_partnet_dist.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
```

#### S3DIS
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    function/evaluate_s3dis_dist.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
```

# Models

|Method | ModelNet40 |
|:---:|:---:|
|Point-wise MLP| [93.2](https://drive.google.com/file/d/1L6WHoDAijkn3r6fvEul5KvHF79KEYxC8/view?usp=sharing) | 
|Pseudo Grid| [93.0](https://drive.google.com/drive/folders/1xBo_rIst6k-69kp6agO2opew3AlYFyjh?usp=sharing) | 
|Adapt Weights| [93.1](https://drive.google.com/file/d/1tu8kO5Fyir1V3Doy-6MguJ58nkPfoIMh/view?usp=sharing) | 
|PosPool| [92.7](https://drive.google.com/file/d/1Mu87SD3VH11nmj85g3uYbIASA3lK8cm4/view?usp=sharing) | 
|PosPool*| [93.2](https://drive.google.com/file/d/1_4o2osPQzM1WQ6QDqkLpwS0m0bsgHVGw/view?usp=sharing) |


We will release other models soon.