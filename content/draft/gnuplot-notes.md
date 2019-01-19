---
title: "Gnuplot Notes"
date: 2018-11-20T19:23:59+08:00
draft: true
tags: [
    "gnuplot"
]
categories: [
    "Development",
]
---

## Gnuplot 可以做什么?

- 画函数曲线

## 要注意什么

- 如何查看帮助
- 如何定义变量, 默认的变量有哪些, 如何定义函数

- 如何设置 x-axis, y-axis range, x/y ticks, x/y label, title, legend,
point/line types, point size, line width, color

- 如何输出: 不同的文件格式

## 如何查看帮助

- `help plot` 查看 plot 的帮助文档

## 如何查看变量

- `show version` 查看 gnuplot 的版本
- `show terminal` 查看当前的 terminal
- `show output` 查看当前的 output 变量
- `show variables` 可以查看当前定义的变量, 包括自定义变量
- `show functions` 可以查看自定义的函数


## 数据来自文件
第一列是 1, 不是从 0 开始计数!

## 表达式
gnuplot 采用 C 语言实现, 因此, `5/2` 的结果是 0 !

```
print 5/2
```

输出 0.

## 如何定义一个变量和函数

```
k = -2
b = 1
f(x) = k*x + b
plot f(x)
```

## 如何定义一个数组

```
array my[100]
do for [i=1:100] {my[i] = cos(i*pi/50)}
plot my title "cos(x)"
```

## 如何从一个文件画图

最简单的, 用

```
plot "xxx.txt" using 1:2
```
使用第一列和第二列. 默认使用 空格分开每列. 空格可以是一个, 也可以是多个

后面加

- `with boxes` 画直方图. gnuplot 不会自动给你分组, 你应该首先分好组, 再调用
- `with lines` 画直线图
- `with dots`
- `with linespoints`
- `with points`

### 对数据做计算

对第二列开方
```
plot "xxx" using 1:( sqrt($2) )
```

求第二列和第三列的平均值
```
plot "xxx" using 1:( ($2 + $3)/2 )
```

求第一列和第二列的 log
```
plot "xxx" using ( (log($1)) ):( (log($2)) )
```


## 如何画一个函数曲线

```
plot sin(x)
```
默认 x 的范围是 -10 到 10, y 的范围刚好覆盖函数的值域. 默认是用实线连接 100
个点来作图.

```
show samples
```
显示当前采样的点数. 默认是 x 为 100; 如果是 3 D 的, 那么 y 的采样率也是 100 个点.

```
set samples 10
plot sin(x)
unset samples
```

`x` 是一个 dummy variable, 所以 `plot sin(x)` 知道 `x` 是一个变量.

使用

```
set dummy t
plot sin(t)
```
来使 `t` 作为 dummy variable


## 使用 NaN

```
f(x) = (x >= 1 || x <= -1) ? x : NaN
plot f(x)
```
可以使得 (-1, 1) 段的值不被画出来!


## 随机数
```
array n[10]
do for [i=1:10] {n[i] = rand(0)}
plot n
```

采用 `help random` 查看相关帮助

## 设置 legend

```
plot sin(x) title "sin(x)",
     cos(x) title "cos(x)"
```

## 设置 label

```
set xlabel "this is xlabel"
set ylabel "this is ylabel"
```

## 设置对数坐标轴

```
set logscale
```
命令设置 x 和 y 轴都用对数坐标

或者分别设置

```
set logscale x
set logscale y
```

要恢复默认值, 则使用
```
unset logscale
unset logscale x
unset logscale y
```

默认是以 10 为底的对数, 使用 `set logscale y 2` 变成以 2 为底.

## 取消 tics
```
unset xtics
unset ytics
```

## 取消 border
```
unset border
```

## 设置 x/y range

```
plot [-2:2] [0:4] 2*x
```
设置 x 的范围为 -2 到 2, y 的显示范围为 0 到 4.

## 设置图片大小
```
set size square
```



## 输出 png 格式

```
set terminal pngcairo
set output "abc.png"
plot sin(x)
```

## 输出 pdf 格式

```
set terminal pdfcairo
set output "abc.pdf"
plot sin(x)
```

画完 `plot` 后, 我们要切换到其他的 terminal 里, 才可以看到生成的pdf!

## 恢复 terminal
当要实时显示所画结果时, 恢复 terminal, 使用
`set terminal` 查看支持的 terminal. 比如
`set terminal aqua`. 或者使用 `unset terminal`, 此时 gnuplot
会自动的选用默认的 terminal.

使用 `show terminal` 来查看当前的 terminal.
