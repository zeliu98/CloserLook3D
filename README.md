# A Closer Look at Local Aggregation Operators in Point Cloud Analysis

By [Ze Liu](https://github.com/zeliu98), [Han Hu](https://github.com/ancientmooner), [Yue Cao](https://github.com/caoyue10), [Zheng Zhang](https://github.com/stupidZZ), [Xin Tong](http://www.xtong.info/)

**Updates**
- July 3, 2020: initial release.

## Introduction

This repo is the official implementation of ["A Closer Look at Local Aggregation Operators in Point Cloud Analysis"](https://arxiv.org/pdf/2007.01294.pdf), which provides clean and the best (to-date) implementations for several representative operators including, **Point MLP based (PointNet++-Like)**, **Pseudo Grid based (KPConv-Like)** and **Adapt Weights (ContinuousConv-Like)**. It also includes a new family of local aggregation operators without learnable weights, named **Position Pooling (PosPool)**, which is simpler than previous operators but performs similarly well or slightly better. Both *PyTorch* and *TensorFlow* implementations are given.

Three datasets are tested, including [ModelNet](https://modelnet.cs.princeton.edu/), [S3DIS](http://buildingparser.stanford.edu/dataset.html) and [PartNet](https://cs.stanford.edu/~kaichun/partnet/). Our implementations all achieve (or are close to) the state-of-the-art accuracy on these benchmarks by proper configurations of each operator type. In particular, one settings achieves `53.8` part category mean IoU on PartNet test set, which outperforms previous best implementations by `7.4 mIoU`.

## Citation

```
@article{liu2020closerlook3d,
  title={A Closer Look at Local Aggregation Operators in Point Cloud Analysis},
  author={Liu, Ze and Hu, Han and Cao, Yue and Zhang, Zheng and Tong, Xin},
  journal={ECCV},
  year={2020}
}
```

## Main Results 

###ModelNet40
|Method | Acc| Tensorflow Model|Pytorch Model|
|:---:|:---:|:---:|:---:|
|Point-wise MLP| 92.8 | [Google](https://drive.google.com/drive/folders/1-CLOVeSmsA-M6sORRhg9mXWRFqloc-n5?usp=sharing) / [BaiDu(wquw)](https://pan.baidu.com/s/1SgQz8Dm561mD9CXMjYOpgw) | [Google](https://drive.google.com/file/d/1L6WHoDAijkn3r6fvEul5KvHF79KEYxC8/view?usp=sharing) / [Baidu(bfa3)](https://pan.baidu.com/s/1mzp4BhMhJwxKwchkdQnD-Q) |
|Pseudo Grid| 93.0 | [Google](https://drive.google.com/drive/folders/16fUdp41jSDD9kHUrXCk_v_1LEkLFsOAp?usp=sharing) / [BaiDu(lvw4)](https://pan.baidu.com/s/1xLavURu0m69BQhrZRaDrEg)  | [Google](https://drive.google.com/drive/folders/1xBo_rIst6k-69kp6agO2opew3AlYFyjh?usp=sharing) / [BaiDu(nslj)](https://pan.baidu.com/s/1cv3RugkiQ9_dGxUIBvCu8Q) |
|Adapt Weights| 93.0 | [Google](https://drive.google.com/drive/folders/1KSVZvPdqTE0I6fceIx7y6Xp1uaxEd5R3?usp=sharing) / [BaiDu(6zrg)](https://pan.baidu.com/s/1byai_J1xi8oSr3iJSyTynw) | [Google](https://drive.google.com/file/d/1tu8kO5Fyir1V3Doy-6MguJ58nkPfoIMh/view?usp=sharing) / [BaiDu(81uf)](https://pan.baidu.com/s/1thUW2gZcuFKzT0yJG8rMEA) |
|PosPool| 92.9 | [Google](https://drive.google.com/drive/folders/1O34VC4APga7hykrNVuoeL8n0LCLNIv96?usp=sharing) / [BaiDu(pkzd)](https://pan.baidu.com/s/1Oo9FsRU5pl6yGy2QpjqIHw) | [Google](https://drive.google.com/file/d/1Mu87SD3VH11nmj85g3uYbIASA3lK8cm4/view?usp=sharing) / [BaiDu(1jp5)](https://pan.baidu.com/s/1jFaWDzYVbbxPSzRVMdtNSg) |
|PosPool*| 93.2 | [Google](https://drive.google.com/drive/folders/10P_Gu5cmaJqg4VyXa27iDi32LH4rpc7o?usp=sharing) / [BaiDu(mjb1)](https://pan.baidu.com/s/1pNBfE2bdmcSY1gG6zQv8Dw) | [Google](https://drive.google.com/file/d/1_4o2osPQzM1WQ6QDqkLpwS0m0bsgHVGw/view?usp=sharing) / [BaiDu(y4xm)](https://pan.baidu.com/s/1Ik07nZguR3KsOtF9pkVHDw) |

###S3DIS
|Method | mIoU | Tensorflow Model |
|:---:|:---:|:---:|
|Point-wise MLP|  66.2 | [Google](https://drive.google.com/drive/folders/1a02JAQnx3WbZ2ngICNSG6fkZ-Mq_knm7?usp=sharing) / [BaiDu(4mhy)](https://pan.baidu.com/s/17wJF7G0FYgMlTzzeJDmFnQ)|
|Pseudo Grid| 65.9 | [Google](https://drive.google.com/drive/folders/1DDkYHliFwlwKkloqBUXke0qP125yXurD?usp=sharing) / [BaiDu(06ta)](https://pan.baidu.com/s/1WekK2KKsccsflKsjSVn6ig) |
|Adapt Weights| 66.5 |[Google](https://drive.google.com/drive/folders/1UtSqxg1Bbfk21Rq7SuLllVbm0rDa3Bli?usp=sharing) / [BaiDu(7w43)](https://pan.baidu.com/s/1tohALjQq771sPLesBEfD2A) |
|PosPool| 66.5 | [Google](https://drive.google.com/drive/folders/15xzFso1lZy-WrAx57eJ7wQd4zYTYPMfz?usp=sharing) / [BaiDu(gqqe)](https://pan.baidu.com/s/19hntRNJYDnQ80RpzqheyTw) |
|PosPool*| 66.7 | [Google](https://drive.google.com/drive/folders/1MT-262K2m65mkUq077DhrFTMqnM-XdEa?usp=sharing) / [BaiDu(qtkw)](https://pan.baidu.com/s/1hvBtQl0ILK18qlyeCOvNMQ) |

### PartNet
|Method | mIoU (val)| mIoU (test) | Tensorflow Model| 
|:---:|:---:|:---:|:---:|
|Point-wise MLP| 48.1 | 51.2 | [Google](https://drive.google.com/drive/folders/13xAG9D6L0bBBXM8kS6pDYEzApOlae7TZ?usp=sharing) / [BaiDu(zw15)](https://pan.baidu.com/s/1jh_Pk44QhXs2m1rCmk5STQ) |
|Pseudo Grid| 50.8 | 53.0 | [Google](https://drive.google.com/drive/folders/1cGdr1vnYB1ZkMjDUfUZ2YCfoscr5NyV0?usp=sharing) / [BaiDu(0mtr)](https://pan.baidu.com/s/114xIbeOzyqOo-vL1z-CVJA) |
|Adapt Weights| 50.1 | 53.5 | [Google](https://drive.google.com/drive/folders/1am3oRLnvj5crHXkLl00gAmdE54vGucPe?usp=sharing) / [BaiDu(551l)](https://pan.baidu.com/s/1S58JVu2IFxphRO1DkNhiOw) |
|PosPool| 50.0 | 53.4 | [Google](https://drive.google.com/drive/folders/1KxjArkFtRUCDkVU8CLO3halL1sU3aZ0N?usp=sharing) / [BaiDu(rb4x)](https://pan.baidu.com/s/1CUyjDucCp7xU5MDt-a2Y9Q) |
|PosPool*| 50.6 | 53.8 | [Google](https://drive.google.com/drive/folders/1eKfuVctpSiAsIpdT0JA3Ns5Su0UhN0QA?usp=sharing) / [BaiDu(2ts3)](https://pan.baidu.com/s/1oE2BJJBw137DJ_D7iFweFw) |

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
