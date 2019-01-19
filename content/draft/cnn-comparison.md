---
title: "Cnn Comparison"
date: 2018-11-28T11:11:12+08:00
draft: true
tags: [
    "Neural_Networks"
]
categories: [
    "Development",
    "Deep_Learning",
]
---


# Introduction
本文对比常见的 CNN 网络结构.

# LeNet

- input 32x32
- c1: convolution, kernel size: 5x5, output size: 28x28, num_output: 6
- tanh
- s2: pooling, output size: 14x14
- c3: convolution, kernel size: 5x5, output size: 10x10, num_output: 10
- tanh
- s4: pooling, output size: 5x5
- c5: convolution, kernel size: 5x5 num_output: 120
- tanh
- f6: full connected, num_output: 84
- output: gaussian connection

# AlexNet

论文pdf[地址][1].

那个著名的图 Figure 2 中,  convolution layer 里包含的东西为

- convolution
- ReLU
- Local response normalization

有两点值得注意:

- 首先, local response normaliation 是在 ReLU 之后. 现在用的 batch normalization
是放在 convolution 后, ReLU 之前的
- 其次, 这里用的是 local response normalization, 现在的网络里好像没见到.
local response normalization 采用了多个 feature map 的信息, 即
`f(n, 0, x, y)`, `f(n, 1, x, y)`, ..., `f(n, c-1, x, y)`. 每个 feature map,
i.e., channel, 只取一个点, 不同 feature map 在相同的位置取一个点.
然后对所去的点做 normalization.
    * 论文里没有说 local response normalization 是采用一张图还是一个batch里的图
    * 注意其与 batch normalization 的不同: (1). batch normalization 采用的是相同的feature map 里的信息, 并且结合了一个batch里所有的图片; (2). batch normalization 中各个 feature map 是独立的做 normalization. (3). Batch normalization 用在 convolution 之后, **但是** 在 ReLU 之前.

AlexNet 中默认是一个 convolution layer 包含了上面三个操作: convolution, ReLU,
local response normalization. 而现在流行的做法是: 一个 convolution layer 就
仅仅做 convolution.

但是: 只有第1个和第2个 convolution layer 包含了 local response normalization,
第3,4,5个没有local response normalization.

第1个和第2个convolutino layer 之后接了 max pooling.

第1个convolution layer 用的 kernel size 为 11x11, stride 为4, num_output 为 96.
图中说input size 为 224x224, 其实是写错了. 应该是 227x227. 作者
在实现的时候肯定是自己补成 227x227, 不然输出不会是 55x55.

第2个convolution layer 用的 kernel size 为 5x5, stride 为1, num_output 为256.
作者应该是默认自动补0了, 以便使得 input size 等于 output size.

第3个convolution layer 用的 kernel size 为 3x3, stride 为1, num_output 为 384.
自动 zero padding.

第4个convolution layer 用的 kernel size 为 3x3, stride 为1, num_output 为 384.

第5个convolution layer 用的 kernel size 为 3x3, stride 为1,

第1个 full connected layer 的num_output 为4096, 后接 ReLU, 再接下一个 full connected layer.

第2个 full connected layer 的num_output 为4096, 后接 ReLU, 再接下一个 full connected.

第3个 full connected layer 的num_output 为1000, 后接 ReLU, 再接 softmax.

总结以下, 有 5 个 convolution layer, 3 个 full connected layer.

用的 loss 为 cross entropy, 作者称它为 multinomial logistic regression.

采用 data augmentation 的方法:

- translation, horizontal reflection
- 从 256x256 中取 224x224.
- 修改 pixel value, 使用的方法是采用 PCA, 现在好像没有见到了. 就先不关注这块了.

采用了 dropout, 但不是 inverted scale dropout.

dropout 只用于第一个full connetected layer 的输出和第二个full connected layer的输出,
keep probability 都为 0.5 .


optimization method:

- momentum 为 0.9
- weight decay 为 0.0005


AlexNet 比 LeNet 多了什么?

- AlexNet 的 convolution layer 更多, 有 5 层, 而 LeNet 只有 3 层
- AlexNet 的 convolution layer 会自动 zero padding, LeNet 里没有自动 padding
- AlexNet 的 convolution layer 用的 kernel size 有 11x11, 5x5, 3x3. 而
LeNet 里都只有 5x5 的 kernel
- AlextNet 第1层的 convolution 的 stride 为 4, LeNet 里所有的 stride 都为 1
- AlexNet 的 convolution layer 的 feature map, 即 channels 更多.
LeNet 最多也才 120.

- LeNet 中, 除了最后一个的 convolution layer 没用 pooling, 其它的都接 pooling.
- AlexNet 中, 第3个和第4个 convolution layer 之间, 没接 pooling


他们的共同点:

- convolution layer 的 num_output 越往后, 越大.

# ZFNet 2013

论文题目为:


[1]: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
