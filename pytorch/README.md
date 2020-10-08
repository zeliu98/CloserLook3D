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

## S3DIS
|Method | mIoU | Model |
|:---:|:---:|:---:|
|Point-wise MLP|  66.3 | [Google](https://drive.google.com/file/d/1WuXb9ajGyE77eAxIrDvQjNzCclWIxf1B/view?usp=sharing) / [Baidu(53as)](https://pan.baidu.com/s/1PRG7sL_Ply_IhuctKQq96g)|
|Pseudo Grid| 65.0 | [Google](https://drive.google.com/drive/folders/1_69Wbe_Au1kLD18zwEoTa3Qa1bLLZY2u?usp=sharing) / [Baidu(8skn)](https://pan.baidu.com/s/1sOI03cbjozsZOKs5Pu7b9w) |
|Adapt Weights| 64.5 |[Google](https://drive.google.com/file/d/1bWGufvua-o1d7P3awaTYnRaUJzYWrizn/view?usp=sharing) / [Baidu(b7zv)](https://pan.baidu.com/s/170hstXHc1eRyVWmRsaHmmg) |
|PosPool| 65.5 | [Google](https://drive.google.com/file/d/12DOacsRjXdyawd_FKS7DQ2LL93ldy-Hm/view?usp=sharing) / [Baidu(z752)](https://pan.baidu.com/s/1Z-0j3flGOFAJbt3v-Hqlcg) |
|PosPool*| 65.5 | [Google](https://drive.google.com/file/d/1KNGOrv4O0kQBzp_BnHfvoy2_T5Xbe-eM/view?usp=sharing) / [Baidu(r96f)](https://pan.baidu.com/s/1YVREDgqAiKZOxcSyXInlRA) |

Data iteration indices: [Google](https://drive.google.com/drive/folders/1BgVEeVcKjs4osqUpdlKqPcyfwkshrHrB?usp=sharing) / [Baidu(m5bp)](https://pan.baidu.com/s/1Aa8vAAbQCwp_IY_jiiiglg)

## PartNet
|Method | mIoU (val)| mIoU (test) | Model| 
|:---:|:---:|:---:|:---:|
|Point-wise MLP| 49.1 | 82.5 | [Google](https://drive.google.com/file/d/19vmcCNitQa-CRSm-5eD2ekrLKb_l57P5/view?usp=sharing) / [Baidu(wxff)](https://pan.baidu.com/s/1ecBICMmGyNpV9QC0DT9RBw) |
|Pseudo Grid| 50.6 | 53.3 | [Google](https://drive.google.com/drive/folders/1qroLeAPSmaSPX_02CEq1LLtbQDQ-lXo6?usp=sharing) / [Baidu(n6b7)](https://pan.baidu.com/s/1eZLCDeyu0Ms8CZFt3CsJmA) |
|Adapt Weights| 50.5 | 52.9 | [Google](https://drive.google.com/file/d/1914kK4DRwdKI8-2wIMNvE4WBEBplpbYr/view?usp=sharing) / [Baidu(pc22)](https://pan.baidu.com/s/10o8aZ96eB9Qwx-IOAvHuMA) |
|PosPool| 50.5 | 53.6 | [Google](https://drive.google.com/file/d/11d-gUIPV2qVIiDT2T6yudQqmJ4Tu-51z/view?usp=sharing) / [Baidu(3qv5)](https://pan.baidu.com/s/1oBi15B4LD_krWYTYr2Kp2A) |
|PosPool*| 51.1 | 53.7 | [Google](https://drive.google.com/file/d/1sOs2T--sw2wT4UXHgVbI5Kr6e1xDxvI0/view?usp=sharing) / [Baidu(czyq)](https://pan.baidu.com/s/19q0Gdpzr6n-i6O3PnKh9SA) |

## ShapeNetPart
|Method | mIoU | msIoU | Acc | Model |
|:---:|:---:|:---:|:---:|:---:|
|Point-wise MLP| 85.7 | 84.1| 94.5 |[Google](https://drive.google.com/file/d/1XLihNmX39zQEoKZ2_qwrxiKzngqTLy_9/view?usp=sharing) / [Baidu(mi2m)](https://pan.baidu.com/s/1MmsQ-m-SIVm2kfgZmp1_Qw)|
|Pseudo Grid| 86.0 | 84.3 | 94.6 |[Google](https://drive.google.com/drive/folders/1qSsj6gmFcn_SElrvZ2OEq6i-Pa1wxC35?usp=sharing) / [Baidu(wde6)](https://pan.baidu.com/s/1Hi20w5j0KfkrTgU6oBgUVQ)|
|Adapt Weights| 85.9 | 84.5 | 94.6 |[Google](https://drive.google.com/file/d/1pjfy3tnnwNO4BV9rXgN82U4njg_YMbSd/view?usp=sharing) / [Baidu(dy1k)](https://pan.baidu.com/s/144VaHNCZHip8Wf-oFaBqUA) |
|PosPool| 85.9 | 84.6 | 94.6 |[Google](https://drive.google.com/file/d/1ca-XO_KEHv9ozB4WoF7sh-p2SPkbnt2I/view?usp=sharing) / [Baidu(r2tr)](https://pan.baidu.com/s/1T41i8m3L8CRF_I_QU3j_QA)|
|PosPool*| 86.2 | 84.8 | 94.8 |[Google](https://drive.google.com/file/d/1Qt3mrxcstKIPidCJqEBAKt5a5zHhn-rW/view?usp=sharing) / [Baidu(27ie)](https://pan.baidu.com/s/1QOWKIoO2cEuvc3b6G2RVWg) |

