# A Closer Look at Local Aggregation Operators in Point Cloud Analysis

By [Ze Liu](https://github.com/zeliu98), [Han Hu](https://github.com/ancientmooner), [Yue Cao](https://github.com/caoyue10), [Zheng Zhang](https://github.com/stupidZZ), [Xin Tong](http://www.xtong.info/).

**Updates**
- July 3, 2020: initial release.

## Introduction

This repo is the official implementation of ["A Closer Look at Local Aggregation Operators in Point Cloud Analysis"](https://arxiv.org/abs/2007.01294), which provides clean and the best (to-date) implementations for several representative operators including, **Point MLP based (PointNet++-Like)**, **Pseudo Grid based (KPConv-Like)** and **Adapt Weights (ContinuousConv-Like)**. It also includes a new family of local aggregation operators without learnable weights, named **Position Pooling (PosPool)**, which is simpler than previous operators but performs similarly well or slightly better. Both *PyTorch* and *TensorFlow* implementations are given.

Three datasets are tested, including [ModelNet](https://modelnet.cs.princeton.edu/), [S3DIS](http://buildingparser.stanford.edu/dataset.html) and [PartNet](https://cs.stanford.edu/~kaichun/partnet/). Our implementations all achieve (or are close to) the state-of-the-art accuracy on these benchmarks by proper configurations of each operator type. In particular, one settings achieves 53.8 part category mean IoU on PartNet test set, which outperforms previous best implementations by 7.4 mIoU.

## Citation

```
@article{liu2019closerlook3d,
  title={A Closer Look at Local Aggregation Operators in Point Cloud Analysis},
  author={Liu, Ze and Hu, Han and Cao, Yue and Zhang, Zheng and Tong, Xin},
  journal={arXiv preprint arXiv:2007.01294},
  year={2020}
}
```

## Main Results (models will come soon)

|Method | ModelNet40 | S3DIS | PartNet (val/test)| 
|:---:|:---:|:---:|:---:|
|Point-wise MLP| 92.8 | 66.2 | 48.1/51.2 |
|Pseudo Grid| 93.0 | 65.9 | 50.8/53.0 |
|Adapt Weights| 93.0 | 66.5 | 50.1/53.5 |
|PosPool| 92.9 | 66.5 | 50.0/53.4 |
|PosPool*| 93.2 | 66.7 | 50.6/53.8 |

**Notes:**
- Overall accuracy for ModelNet40, mean IoU for S3DIS with Area-5, mean part-category IoU for PartNet are reported.
- `Point-wise MLP` denotes `PointNet++-like` operators.
- `Pseudo Grid` denotes `KPConv-like` operators.
- `Adapt Weights` denotes `ContinuousConv-like` operators.
- `PosPool` is a new parameter-free operator.
- `PosPool*` denotes the sin/cos embedding variant of `PosPool` (see description in the paper).

## Install
- For `tensorflow` users, please refer to [README.md](./tensorflow/README.md) for more detailed instructions. Our main experiments are conducted using this code base.
- For `pytorch` users, please refer to [README.md](./pytorch/README.md) for more detailed instructions.

## Acknowledgements

Our `tensorflow` codes borrowed a lot from [KPCONV](https://github.com/HuguesTHOMAS/KPConv).

## License

The code is released under MIT License (see LICENSE file for details).
