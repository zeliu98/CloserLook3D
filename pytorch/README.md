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

**Part Segmentation on ShapeNetPart**

You can download ShapeNetPart dataset from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip) (635M). Unzip and move (or link) it to `data/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0`.

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
│   ├── ShapeNetPart
│   │   └── shapenetcore_partanno_segmentation_benchmark_v0
│   │       ├── README.txt
│   │       ├── synsetoffset2category.txt
│   │       ├── train_test_split
│   │       ├── 02691156
│   │       ├── 02773838
│   │       ├── 02954340
│   │       ├── 02958343
│   │       ├── 03001627
│   │       ├── 03261776
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

#### ShapeNetPart
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node <num_of_gpus_to_use> \
    function/train_shapenetpart_dist.py --cfg <config file> [--log_dir <log directory>]
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

#### ShapeNetPart
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    function/evaluate_shapenetpart_dist.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
```

#### S3DIS
```bash
python -m torch.distributed.launch --master_port <port_num> --nproc_per_node 1 \
    function/evaluate_s3dis_dist.py --cfg <config file> --load_path <checkpoint> [--log_dir <log directory>]
```

# Models

## ModelNet40
|Method | Acc | Model |
|:---:|:---:|:---:|
|Point-wise MLP| 93.0 |[Google](https://drive.google.com/file/d/15O_W7gxgO8JbzduAQEXd4hHvSh5cYRA9/view?usp=sharing) / [Baidu(fj13)](https://pan.baidu.com/s/1GmBNCTeyWoE7ISKsnSqlJA)| 
|Pseudo Grid| 93.1 |[Google](https://drive.google.com/drive/folders/1ZYG_jIUWXcyf-HuAH-zT-QUIZaIgwjhv?usp=sharing) / [Baidu(gmh5)](https://pan.baidu.com/s/1JZDIZGnZZvzMac5bkuMGng)| 
|Adapt Weights| 92.9 |[Google](https://drive.google.com/file/d/1ZxLi0loYV3tdaBgbuJfHXqtknMFmLjh1/view?usp=sharing) / [Baidu(bbus)](https://pan.baidu.com/s/1yS9RfdQtCHNsIkKGrfeDFg)| 
|PosPool| 93.0 |[Google](https://drive.google.com/file/d/1j9_JqxVPEsRjhOMQUeTxmyhJQ9zb65RC/view?usp=sharing) / [Baidu(wuuv)](https://pan.baidu.com/s/1tTjFEIhfqrttRxb32h2URQ)| 
|PosPool*| 93.3 |[Google](https://drive.google.com/file/d/1HSu6K-prMka4tnjx6pMh82oy2igbKzCV/view?usp=sharing) / [Baidu(qcc6)](https://pan.baidu.com/s/1vtwsqdCYUXKiMBc240JhqA)|

## ShapeNetPart
|Method | mIoU | msIoU | Acc | Model |
|:---:|:---:|:---:|:---:|:---:|
|Point-wise MLP| 85.7 | 84.1| 94.5 |[Google](https://drive.google.com/file/d/1XLihNmX39zQEoKZ2_qwrxiKzngqTLy_9/view?usp=sharing) / [Baidu(mi2m)](https://pan.baidu.com/s/1MmsQ-m-SIVm2kfgZmp1_Qw)|
|Pseudo Grid| 86.0 | 84.3 | 94.6 |[Google](https://drive.google.com/drive/folders/1qSsj6gmFcn_SElrvZ2OEq6i-Pa1wxC35?usp=sharing) / [Baidu(wde6)](https://pan.baidu.com/s/1Hi20w5j0KfkrTgU6oBgUVQ)|
|Adapt Weights| 85.9 | 84.5 | 94.6 |[Google](https://drive.google.com/file/d/1pjfy3tnnwNO4BV9rXgN82U4njg_YMbSd/view?usp=sharing) / [Baidu(dy1k)](https://pan.baidu.com/s/144VaHNCZHip8Wf-oFaBqUA) |
|PosPool| 85.9 | 84.6 | 94.6 |[Google](https://drive.google.com/file/d/1ca-XO_KEHv9ozB4WoF7sh-p2SPkbnt2I/view?usp=sharing) / [Baidu(r2tr)](https://pan.baidu.com/s/1T41i8m3L8CRF_I_QU3j_QA)|
|PosPool*| 86.2 | 84.8 | 94.8 |[Google](https://drive.google.com/file/d/1Qt3mrxcstKIPidCJqEBAKt5a5zHhn-rW/view?usp=sharing) / [Baidu(27ie)](https://pan.baidu.com/s/1QOWKIoO2cEuvc3b6G2RVWg) |


We will release other models soon.