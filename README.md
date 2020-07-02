# A Closer Look at Local Aggregation Operators in Point Cloud Analysis
By [Ze Liu](https://github.com/zeliu98), [Han Hu](https://github.com/ancientmooner), [Yue Cao](https://github.com/caoyue10), [Zheng Zhang](https://github.com/stupidZZ), [Xin Tong](http://www.xtong.info/).

## Introduction
Recent advances of network architecture for point cloud processing are mainly driven by new designs of local aggregation operators. However, the impact of these operators to network performance is not carefully investigated due to different overall network architecture and implementation details in each solution. Meanwhile, most of operators are only applied in shallow architectures. In this paper, we revisit the representative local aggregation operators and study their performance using the same deep residual architecture. Our investigation reveals that depsite the different designs of these operators, all of these operators make surprisingly similar contributions to the network performance under the same network input and feature numbers and result in the state-of-the-art accuracy on standard benchmarks. This finding stimulate us to rethink the necessity of sophisticated design of local aggregation operator for point cloud processing. To this end, we propose a simple local aggregation operator without learnable weights, named Position Pooling (PosPool), which performs similarly or slightly better than existing sophisticated operators. In particular, a simple deep residual network with PosPool layers achieves outstanding performance on all benchmarks, which outperforms the previous state-of-the methods on the challenging PartNet datasets by a large margin (7.4 mIoU). 
## Install
- For `tensorflow` users, please refer to [README.md](./tensorflow/README.md) for more detailed instructions. Our main experiments are carried on this code base.
- For `pytorch` users, please refer to [README.md](./pytorch/README.md) for more detailed instructions.

## Performances

|Method | ModelNet40 | S3DIS | PartNet (val/test)| 
|:---:|:---:|:---:|:---:|
|Point-wise MLP| 92.8 | 66.2 | 48.1/51.2 |
|Pseudo Grid| 93.0 | 65.9 | 50.8/53.0 |
|Adapt Weights| 93.0 | 66.5 | 50.1/53.5 |
|PosPool| 92.9 | 66.5 | 50.0/53.4 |
|PosPool*| 93.2 | 66.7 | 50.6/53.8 |

We report overall accuracy for ModelNet40, mean IoU for S3DIS with Area-5, mean part-category IoU for PartNet.

***Note***: `PosPool*` means the sin/cos embedding variant.

## Acknowledgements

Our `tensorflow` codes borrowed a lot from [KPCONV](https://github.com/HuguesTHOMAS/KPConv).

## License

The code is released under MIT License (see LICENSE file for details).

## Updates

- July 2, 2020: Initial release.