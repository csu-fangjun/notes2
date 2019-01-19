---
title: "Gradient Descent"
date: 2018-11-17T13:52:19+08:00
draft: true
tags: [
    "Gradient_Descent"
]
categories: [
    "Development",
    "Optimization",
    "Deep_Learning",
]
---

## Introduction
本文介绍了 change of basis 对 gradient descent 及 Newton's method 的影响.

## 一个例子

对于函数 $f(x) = x^2$, 其中 $x \in \mathbb{R}$, 使用 gradient descent
求其最小值.

假设初值 $x_0 = 1$, $f'(x) = 2x$, 假设 step size (i.e., learning rate) 为 0.1

- $x_0 = 1$, $f(x_0) = 1$
- $x_1 = x_0 - 0.1\times 2 \times f'(x_0) = 1 - 0.1 \times 2 \times 1 = 0.8$, $f(x_1) = 0.64$
- $x_2 = 0.8 - 0.1 \times 2 \times 0.8 = 0.64$, $f(x_2) = 0.4096$
- $x_3 = 0.64 - 0.1 \times 2 \times 0.64 = 0.512$, $f(x_3) = 0.262144$

如果令 $z = 2x$, 即 $x = \frac{z}{2}$,
那么 $g(z) = f(\frac{z}{2}) = \frac{z^2}{4}$. 求 $f(x)$
的最小值, 等价于求 $g(z)$ 的最小值.

假设初值 $z = 2$, $g'(z) = \frac{z}{2}$, step size 为 0.1

- $z_0 = 2$, $g(z_0) = 1$
- $z_1 = 2 - 0.1 \times \frac{2}{2} = 1.9$, $g(z_1) = 0.90$
- $z_2 = 1.9 - 0.1 \times \frac{1.9}{2} = 1.805$, $g(z_2) = 0.8145$

从以上的例子可看出, change of variable (i.e. change of basis) 对
gradient descent 的性能有非常大的影响. $f(x)$ 下降的明显比
$g(z)$ 快.

## Change of basis
对于 $f(x) \in \mathbb{R}$, $x \in \mathbb{R}^n$, 在 $x$ 处用一阶 taylor 代替它
$$
f(x+\delta x) = f(x) + \nabla ^\top f(x) \cdot \delta x
$$

注意:
$$
\frac{\partial f(x)}{\partial x} = \nabla ^\top f(x) \in \mathbb{R} ^ {1 \times n}
$$

在 $x$ 处, 我们希望找一个 $\delta x$, 使得 $f(x +\delta x)$
的值比 $f(x)$ 小. 由于上面已经用一阶 taylor 代替了 $f(x + \delta x)$,
因此, 我们只需要 $\nabla^\top f \cdot \delta x \lt 0$ 即可以.

只要 $\delta x$ 与 $\nabla f$ 的夹角度大于$90^\circ$, 我们都有
$\nabla^\top f \cdot \delta x \lt 0$, 一般都取 $\delta x = -\alpha \cdot\nabla f$,
其中 $\alpha \gt 0$ 为 step size. 由于上面使用 一阶 taylor 去代替 $f(x+\delta x)$,
因此 $\delta x$ 必须足够的小, 不然线性近似的误差就很大了. 所以 $\alpha$
一般都取得比较小.

如何取 $\alpha$ 的值, 有很多的方法, 调研总结后再展开.

gradient descent 的更新公式为
$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

终止条件一般为:

- $x_{k+1}$ 与 $x_k$ 的差值小于某个 threshold
- 或者, 指定的迭代次数已到达
- 或者, $f(x_k)$ 与 $f(x_{k+1})$ 的差值小于某个 threshold

如果令
$$
z = Bx
$$

那么
$$
g(z) = f(x) = f(B^{-1} z)
$$

$$
\frac{\partial g(z)}{\partial z} =
\frac{\partial f(x)}{\partial x} \frac{\partial x}{\partial z} =
\frac{\partial f(x)}{\partial x} B^{-1}
$$

$$
\nabla g(z) = (\frac{\partial g(z)}{\partial z})^\top
= B^{-\top} \nabla f(x)
$$

基于 $x$ 的更新公式为
$$
x_{k+1} = x_k - \alpha \nabla f(x) \
$$

基于 $z$ 的更新公式为

$$
z_{k+1} = z_k - \alpha \nabla g(z)
$$

即
$$
B x_{k+1} = B x_k - \alpha B^{-\top} \nabla f(x)
$$

化简后可得
$$
x_{k+1} = x_k - \alpha B^{-1}B^{-\top} \nabla f(x)
$$

因此 basis 变换之后, 在 $z$ 里下降的速度和在原来 $x$
里下降的速度是不一样的.

## Newton's Method

在 $f(x+\delta x)$ 处, 用 taylor 二阶去代替它
$$
f(x+\delta x) = f(x) + \nabla^\top f(x) \delta x + \frac{1}{2} \delta x^\top H \delta x
$$

因为我们要最小化 $f(x+\delta x)$, 由于此时它是一个二次函数, 令它关于$\delta x$
的导数为0, 可得
$$
\frac{\partial f(x+ \delta x)}{\delta x} =
\nabla^\top f(x) + \delta x^ \top H = 0
$$

即
$$
\delta x = -H^{-1} \nabla f(x)
$$

Newton's method 的更新公式为
$$
x_{k+1} = x_k + \delta x = x_k - H^{-1} \nabla f(x)
$$

注意, 这里没有 step size !!!

Change basis 对 Newton's method 有何影响呢?

还是令 $z = B x$, 从上面可知
$$
\frac{\partial g(z)}{\partial z} =
\frac{\partial f(x)}{\partial x} \frac{\partial x}{\partial z} =
\frac{\partial f(x)}{\partial x} B^{-1}
$$

$$
\mathrm{Hessian}(z) =
\frac{(\frac{\partial g(z)}{\partial z})^\top}{\partial z}
= \frac{\nabla g(z)}{\partial z}
$$

$$
\mathrm{Hessian}(z) =
B^{-\top} \frac{(\frac{\partial f(x)}{\partial x})^\top}{\partial z}=
B^{-\top} \frac{(\frac{\partial f(x)}{\partial x})^\top}{\partial x}\frac{\partial x}{\partial z}
$$

$$
\mathrm{Hessian}(z) =
B^{-\top} \mathrm{Hessian}(x)\frac{\partial x}{\partial z} =
B^{-\top}\mathrm{Hessian}(x) B^{-1}
$$

$$
\mathrm{Hessian}(z)^{-1} =
B\, \mathrm{Hessian}(x)^{-1} B^\top
$$

在 $z$ 里的更新公式为

$$
z_{k+1} = z_k - \mathrm{Hessian}(z)^{-1} \nabla g(z)
$$

变换到 $x$, 得
$$
B x_{k+1} = B x_k - B H^{-1} B^\top \nabla g(z) =
B x_k - B H^{-1} B^\top  B^{-\top} \nabla f(x) = B x_k - B H^{-1} \nabla f(x)
$$

即
$$
x_{k+1} = x_k - H^{-1} \nabla f(x)
$$

所以, change of basis 对 Newton's method 没有影响, 在 $z$ 里走一步等价于
在原来的 $x$ 里走对应的一步. 也就是说, 从 $x$ 里要经过 $N$ 不才能到达
最优点, 那么从$z$里也要走 $N$ 步才能到达最优点.

## References and tutorials

- [Convex Optimization: Fall 2013][1], at CMU
    * lecture slides and assignment sheet/code are available online
    * there are also lecture videos on YouTube !

- [Lecture Notes: Some notes on gradient descent][2], pdf, from Prof. Toussaint

- [lecture notes: Empirical Risk Minimization and Optimization][3], 2010, statistic
machine learning, 

[3]: https://people.cs.umass.edu/~domke/courses/sml2010/02optimization.pdf
[2]: https://www.cs.virginia.edu/yanjun/teach/2015f/lecture/L4-GD.pdf
[1]: http://www.stat.cmu.edu/~ryantibs/convexopt-F13/
