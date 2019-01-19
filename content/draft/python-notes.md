---
title: "Python Notes"
date: 2018-11-01T20:37:23+08:00
draft: true
tags: [
    "Python"
]
categories: [
    "Development",
]
---

<https://nbviewer.jupyter.org/>

## Setup

```sh
pip install virtualenv
brew install python3
brew link python
```

```sh
virtualenv --no-site-packages -p python3 venv
```
which creates a directory `venv` inside the current directory.

Activate the virtual environment
```sh
source venv/bin/activate
```

to exit the virtual environment, run

```sh
deactivate
```


To install libs inside the virtual environment,
```sh
pip install --upgrade pip
pip install jupyter
pip install numpy
pip install scipy
```

To run a notebook
```sh
jupyter notebook path/to/xxx.ipynb
# or
jupyter notebook
```

To setup vim for jupyter notebook, refer to <https://github.com/lambdalisue/jupyter-vim-binding>

```sh
# Create required directory in case (optional)
mkdir -p $(jupyter --data-dir)/nbextensions
# Clone the repository
cd $(jupyter --data-dir)/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
# Activate the extension
jupyter nbextension enable vim_binding/vim_binding
```

Then copy <https://github.com/csukuangfj/kfj-vim/blob/dev/jupyter/custom.js>
to `~/.jupyter/custom/custom.js`.

## Numpy

The most important command is
```
import numpy as np
np.info(np.max)
```

`np.info()` to look for help!

### axis in numpy
We have `a.shape==(2, 3, 4)`:

- `np.sum(a, axis=0)`, the return shape is `(3, 4)`, that is, we merge two
planes of size `(3, 4)` and only keep the sum.

```python
a = np.arange(24).reshape(2, 3, 4)
print(a)
```

```
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

```python
print(np.sum(a, axis=0))
```

```
# sum
[[12 14 16 18]
 [20 22 24 26]
 [28 30 32 34]]
```

For the `axis=0`, we overlay
```
0 1 2   3     12 13 14 15
4 5 6   7   + 16 17 18 19
8 9 10 11     20 21 22 23
```

- `np.sum(a, axis=1)`, the return shape is `(2, 4)`;
it processes every plane independently. For every plane of
shape `(3, 4)`, we merge rows to change it to the shape`(4,)`
and keep the sum.

```python
print(np.sum(a, axis=1))
[[12 15 18 21]
 [48 51 54 57]]
```

- `np.sum(a, axis=2)` returns a shape of size `(2, 3)`.
Every plane is processed independently and to change `(3, 4)`
to `(3,)`, we merge columns of every plane.

```python
print(np.sum(a, axis=2))
[[ 6 22 38]
 [54 70 86]]
```



### References
- [100 numpy exercises][1]


[1]: http://www.labri.fr/perso/nrougier/teaching/numpy.100/
