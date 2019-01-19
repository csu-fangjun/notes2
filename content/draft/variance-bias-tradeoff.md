---
title: "Variance Bias Tradeoff"
date: 2018-11-14T10:00:46+08:00
draft: true

tags: [
    "Machine_Learning"
]
categories: [
    "Development",
]
---

# 如何理解 Variance 与 Bias

首先要弄清楚它们俩是如和通过**数学公式**来定义的.

给定一系列数据
$$
(x_i, y_i), \;\;i=1,2,\cdots,n, \;\;\; x_i \in \mathbb{R}^m, \;\; y_i \in \mathbb{R}
$$

找一个函数
$$
f : \mathbb{R^m} \rightarrow \mathbb{R}
$$
使得 $f(x_i)$ 尽可能的接近 $y_i$, 即要使如下  $\mathrm{loss}$ 最小.

$$
\mathrm{loss} = \frac{1}{n} \sum_{i=1}^n (f(x_i) - y_i)^2
$$

上述的 $\mathrm{loss}$ 就是和 variance-bias 密切相关的.

**假设**: 所有的$(x_i, y_i)$ 都是从某一概率分布独立取样的, 即
$(x_i, y_i)$ 是 **i.i.d.**, identical and independent distributed.

给定$x$的情况下, $y$ 的平均值为 $\mathbb{E}[y|x]$, 以下记为 $\mathbb{E}[y]$.

给定$x$的情况下, 通过函数 $f$ 预测出来的平均值为 $\mathbb{E}[f(x)]$, 以下
记为 $\mathbb{E}[f]$.

上述 $loss$ 的期望为
$$
\mathbb{E}[(f - y)^2]
$$

Variance 的定义为

$$
\mathbb{E}[(f - \mathbb{E}(f))^2]
$$

给定一系列的 $x_i$, 我们都可以求得 $f(x_i)$, 也即可以得到 $\mathbb{E}(f)$,
从而可以得到上述定义的 Variance. 

如果 $f(x)$ 是一个常数, 即对每一个输入 $x$,
$f(x)$ 都预测出同一个值, 那么方差就为 0 了.

$f(x)$ 震荡的越厉害, 那么它的方差也就越大. 因此, 二次函数的 Variance
就比一次函数的大. 增加模型的复杂度, 就会增加 Variance.

在 training 的时候, 如果我们选用一个非常非常复杂的模型让 training error
趋于0, 那么由于模型太过复杂, 会造成 Variance 非常的大.

Variance 过大, 说明有 **overfitting** 的可能性.

如果模型选的太简单, 虽然可以可以使 Varaince 变小, 例如, 选一个 constant model,
可以使 Variance 变为0, 但此时就会存在 **underfitting** 的可能性.

一个例子就是 $K$ nearest neighbor 中的 $K$ 的选取问题. 如果 $K$
等于所有的数据量, 那么 $f$ 的预测输出就是一个常量, 此时 variance 为0.
但 bias 却很大. 如果 $K$ 为 1, 那么预测输出的变化就会非常大, 此时
variance 就很大. 类似的例子为 Parzen window 中的 $\sigma$ 的选取.

Bias 的定义为
$$
\mathbb{E}[y - f(x)] = \mathbb{E}[y] - \mathbb{E}[f]
$$
直观上的理解 Bias: 预测出来的值 $f(x)$ 与真实值 $y$ 之间的差的平均.

**注意**: Bias 的定义里没有平方, 而 Variance 里是有平方的, 也就是说他们
的单位不一样! $\mathrm{loss}$ 的定义里也是平方!

如果我们选一个复杂的模型使得对任意 $x$, $f(x)$ 都接近 $y$, 那么我们就可以
得到一个很低的 bias. 但前文以说过, 复杂的模型会提高 Variance.

## Decomposition

首先看看
$$
\mathrm{Bias}^2 + \mathrm{Variance}
$$

$$
\begin{align}
(\mathbb{E}[y] - \mathbb{E}[f])^2 + \mathbb{E}[(f - \mathbb{E}[f])^2]=
\mathbb{E}[y]^2 - 2\mathbb{E}[y]\mathbb{E}[f] + \mathbb{E}[f]^2 +
\mathbb{E}[f^2] -\mathbb{E}[f]^2
\end{align}
$$

化简后, 得到

$$
\begin{align}
(\mathbb{E}[y] - \mathbb{E}[f])^2 + \mathbb{E}[(f - \mathbb{E}[f])^2]=
\mathbb{E}[y]^2 - 2\mathbb{E}[y]\mathbb{E}[f] + \mathbb{E}[f^2]
\end{align}
$$

然后看看 $\mathrm{loss}$ 的展开

$$
\mathrm{loss} = \mathbb{E}[(y - f)^2] = \mathbb{E}[y^2 - 2yf + f^2]
= \mathbb{E}[y^2] - 2\mathbb{E}[y]\mathbb{E}[f] + \mathbb{E}[f^2]
$$

上述两式的差别就在于 $\mathbb{E}[y]^2$ 于 $\mathbb{E}[y^2]$.

我们知道
$$
\mathbb{E}[y^2] = \mathbb{E}[y]^2 + \mathbb{E}[(y - \mathbb{E}[y])^2]
$$


因此,
$$
\mathrm{loss} = \mathrm{Bias}^2 + \mathrm{Variance} + \mathbb{E}[(y - \mathbb{E}[y])^2]
$$

因为我们无法得到 $\mathbb{E}[(y - \mathbb{E}[y])^2]$, 所以我们
只有操作 $\mathrm{Bias}$ 与 $\mathrm{Variance}$ 来减少 $\mathrm{loss}$.

**注意**: 上述的 $\mathrm{loss}$ 是与 $y$ 相关的, 因此 Variance Bias Decomposition
只适用于 $Supervised$ Learning.

另外一种推导方法[^1]:

$$
\mathbb{E}[(y - f)^2] =
\mathbb{E}[((y - \mathbb{E}[y]) + (\mathbb{E}[y] - f))^2] =
\mathbb{E}[(y - \mathbb{E}[y])^2 + (\mathbb{E}[y] - f)^2 - (y-\mathbb{E}[y])(\mathbb{E}[y] - f)]
$$

化简, 得到
$$
\mathbb{E}[(y - f)^2] =
\mathbb{E}[(y - \mathbb{E}[y])^2] + \mathbb{E}[(\mathbb{E}[y] - f)^2]
- \mathbb{E}[(y-\mathbb{E}[y])]\mathbb{E}[(\mathbb{E}[y] - f)]
$$

因为
$$
\mathbb{E}{y - \mathbb{E}[y]} = \mathbb{E}[y] - \mathbb{E}[y] = 0
$$

所以
$$
\mathbb{E}[(y - f)^2] =
\mathbb{E}[(y - \mathbb{E}[y])^2] + \mathbb{E}[(f - \mathbb{E}[y])^2]
$$

其中, $\mathbb{E}[(y - \mathbb{E}[y])^2]$ 为 $y$ 自己的方差.

Bias-Variance-Tradeoff 就是对 $\mathbb{E}[(f - \mathbb{E}[y])^2]$
的分解.

$$
\mathbb{E}[(f - \mathbb{E}[y])^2] =
\mathbb{E}[((f - \mathbb{E}[f]) + (\mathbb{E}[f] - \mathbb{E}[y]))^2]
$$

同理, 可得
$$
\mathbb{E}[(f - \mathbb{E}[y])^2] =
\mathbb{E}[(f - \mathbb{E}[f])^2] + (\mathbb{E}[f] - \mathbb{E}[y])^2
$$

**解释**:

- 我们想要最小化 $\mathbb{E}[(f - \mathbb{E}[y])^2]$, 即 预测值偏离
所有真值的累计.
- 引入一个中间变量 $\mathbb{E}[f]$
- 我们的目标等价与最小化 $\mathbb{E}[(f - \mathbb{E}[f])^2]$
与 $(\mathbb{E}[y] - \mathbb{E}[f])^2$ 之和
- 其中 $\mathbb{E}[(f - \mathbb{E}[f])^2]$ 为 Variance
- $(\mathbb{E}[y] - \mathbb{E}[f])^2$ 为 $\mathrm{Bias}^2$

因此, Bias 的解释就是预测的**平均值**与真实的**平均值**
之间还差多少. 若偏差为0, 就说 $f$ 是 $y$ 的一个 unbiased estimator.
Bias 解释了模型与真实值之间的关系.

而 Variance 是对模型本身的解释. 对模型所有的输出求平均值, 然后看
单个的预测值偏离这个平均值多大.

Regularization 可以减少 Variance, 因为它降低了模型的复杂度.
一般使用 cross validation 来确定 regularization 使用的 $\lambda$.
Regularization 也可以理解为对模型做 smoothing, 这样的化,
模型曲线变光滑了, 因此 Variance 也随之减少了.


## 总结

1. 模型太复杂, 使得$f(x)$ 过于震荡, 从而造成 variance, 即 $\mathbb{E}[(f - \mathbb{E}[f])^2]$ 太大.
同时由于复杂的模型可以使我们更好的接近 $y$, 从而可以降低 bias, 即
$\mathbb{E}[y] - \mathbb{E}[f]$.
    * 复杂的模型, high variance, low bias, overfitting
2. 模型太简单, $f(x)$ 非常平坦, variance 非常的小; 极端情况下, 当 $f(x)$
为一常数时, variance 为0. 由于模型太简单, 导致我们不能很好的逼进 $y$,
使得 bias, 即 $\mathbb{E}[y] - \mathbb{E}[f]$ 过大.
    * 简单的模型, low variance, high bias, underfitting

3. 所以存在一个 tradeoff, 这个 tradeoff 就是通过 variance 与 bias 来体现的.
4. 只适用于 supervise learning

# How to compute bias and variance

Refer to page 22 of <http://www.cs.cmu.edu/~wcohen/10-601/bias-variance.pdf>.

We have
$$
x_i, y_i, f(x_i), \;\; i = 1, 2, \cdots, n
$$

- To compute the variance: compute the variance of $f(x_i)$
- To compute the bias: compute $\frac{1}{n}\sum_i (f(x_i) - y_i)$


# 例子
## 1

下面两张图来自 [^1]. 注意 total error, bias 与 variance 的变化趋势.

total error 先减少, 再增大. bias 与 variance 一般都是单调的.

![][2]
![][3]

## 2
下图来自 [^5]. 其中的 $K$ 为 total error.

注意模型复杂度与 bias, variance 的关系. Bias 可正可负, 所以图中取平方.
同时, total error 也等于 bias 的平方与 variance 之和.

## 3
一般来说, variance 增加, 则 bias 减少; variance 减少, 则 bias 增加.

但如果算法**有问题**, 则 variance 增加, biase **也会** 增加!


![][4]




[^5]: https://people.cs.umass.edu/~domke/courses/sml2010/01overfitting.pdf , lecture notes from "Overfitting, Model Selection, Cross Validation, Bias-Variance"
[4]: /images/variance-bias-decomposition/curve.png
[3]: /images/variance-bias-decomposition/parzen-window.png
[2]: /images/variance-bias-decomposition/knn.png

[^1]: paper(1992): Neural networks and the bias/variance dilemma, http://web.mit.edu/6.435/www/Geman92.pdf . It is the first paper that proposes variance bias decomposition.
