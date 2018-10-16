---
title: "Python Notes"
date: 2018-11-01T20:37:23+08:00
draft: true
---

<https://nbviewer.jupyter.org/>

# Setup

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
```
