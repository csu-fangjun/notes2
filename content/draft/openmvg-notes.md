---
title: "OpenMVG Notes"
date: 2018-11-09T21:32:31+08:00
draft: true
tags: [
    "openmvg"
]
categories: [
    "Development",
    "Compute_Vision",
]
---

```sh
git clone https://github.com/openMVG/openMVG.git
cd openMVG
git submodule init
git submodule update

cd ..
mkdir openMVG_Build
cd openMVG_Build

cmake \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$HOME/software/openmvg \
-DOpenMVG_BUILD_SHARED=ON \
-DOpenMVG_BUILD_TESTS=OFF \
-DOpenMVG_BUILD_DOC=OFF \
-DOpenMVG_BUILD_EXAMPLES=ON \
-DOpenMVG_BUILD_OPENGL_EXAMPLES=ON \
-DOpenMVG_BUILD_SOFTWARES=ON \
-DOpenMVG_BUILD_GUI_SOFTWARES=ON \
-DOpenMVG_BUILD_COVERAGE=OFF \
-DOpenMVG_USE_OPENMP=ON \
\
-DOpenMVG_USE_OPENCV=ON \
-DOpenMVG_USE_OCVSIFT=ON \
../openMVG/src/
```

# Third Party Libraries

## Linear programming
It uses

- <https://projects.coin-or.org/Clp>
- <https://projects.coin-or.org/Osi/>
