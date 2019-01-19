---
title: "Neural Networks"
date: 2018-11-14T09:09:58+08:00
draft: true
tags: [
    "Neural_Networks"
]
categories: [
    "Development",
    "Deep_Learning",
]
---

# Autodiff
- [Automatic Differentiation and Neural Networks][1]
    * it constructs a 1 hidden layer with 72 neurons for MNIST data set
and displays the weights and intermediate results in an image form, which
is good for understanding.
    * it shows a schematic diagram illustrating the back propagation procedure
    * it does not show how to implement the auto differentiation
    * 这篇博客[^2]也举了同样一个例子, 不过, 这里的图更好看. 一看就是 tikz 画的图.

Figure 2 from this paper [^3]

autograd 一个软件

chainer 一个软件

A Hitchhiker’s Guide to Automatic Differentiation, 2016
https://arxiv.org/pdf/1411.0583.pdf


an example code implementation
https://github.com/modsim/CADET-semi-analytic/blob/master/ThirdParty/FADBAD%2B%2B/fadiff.h

# Optimizers

To read
- [Optimization methods for large-scale machine learning][4], paper from arxiv, 2016


[4]: https://arxiv.org/abs/1606.04838
[^3]: https://arxiv.org/pdf/1502.05767.pdf , "Automatic Differentiation in Machine Learning: a Survey" gives a good description of how to implement autodiff for a simple function
[^2]: https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/
[1]: https://people.cs.umass.edu/~domke/courses/sml2010/07autodiff_nnets.pdf


