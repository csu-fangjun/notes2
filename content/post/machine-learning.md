---
title: "Machine Learning"
date: 2018-10-25T16:13:09+08:00
draft: false
tags: [
    "machine_learning",
]
categories: [
    "Development",
]
---


 - murphy-gaussians: <http://ais.informatik.uni-freiburg.de/teaching/ws17/mapping/pdf/murphy-gaussians.pdf>

- [Masterpraktikum, Deep Learning Lab][33], Uni Freiburg, WS 2018/19
    * code at github: <https://github.com/aisrobots/dl-lab-2018>
- [Deep Learning Course, Control Section][37], Uni Freiburg, WS16/17
    * code at github: <https://github.com/mllfreiburg/dl_lab_2016>
- [CS231n: Convolutional Neural Networks for Visual Recognition][38], Uni Stanford, 2018
- a book: [Neural Networks and Deep Learning][39]
    * free online html
- [CS224n: Natural Language Processing with Deep Learning][40], stanford, 2018


- [The Projective Camera][46], note that the pinhole model is only
a special class of the projective camera model

- [The Perspective Camera][47], i.e., the pinhole model
- [The Weak-Perspective Camera][48], it is similar to the pinhole camera model except that it groups objects in the space with similar depth and replace their depth
with the same value $z$. Thus, during the projection, $\frac{f}{z}$ is the same
for all pixels, which is similar to orthographic projection. When to use this model:
when the object is far away from the camera such that $z >> f$ and the field of view
is small. Parallel lines are preserved during projection.
- [The orthographic camera][49]

### Camera Calibration

Refer to <http://www.cse.iitd.ernet.in/~suban/vision/geometry/node39.html>

- [Tsai camera model and calibration][51]
- [Camera calibration and absolute conic][52]
- [Camera calibration and absolute conic][55]
- [What does calibration give?][54], angle between rays, normal vector
of a plane through the camera center.
- [The image of the absolute conic][56]
- [A simple calibration device][57], using absolute conic for calibration.
It needs 5 image of circular points, which are obtained from 3 planes.
- using vanishing points to determine the absolute conic, refer to
the lecture slide [here][58]


In the orthographic camera model, every object has the same magnification;
while in the weak perspective model, distant object looks smaller. Objects have
similar $z$ have the same magnification factor.

Weak perspective projection can be considered as a combination of perspective and orthographic
projection.

Refer to [Simplified Camera Projection Models][50], pdf.

### Projective Geometry

- ideal points, points at infinity
- line at infinity, plane at infinity
- circular points, absolute conic
- vanishing points
    * if a point lies on the plane at infinity, then its image is called vanishing point
    * two parallel lines intersect at a point on the plane at infinity, so the image
of the intersection is a vanishing point
    * it can be used for camera calibration! (1) identify two vanishing points;
(2) identify the angle of the two corresponding lines; (3) we get a constraint;
(4) the intrinsics have 5 degree of freedom, so we need to find five pairs of vanishing points
    * by vanishing points, we can use just one image to calibrate an image!
- vanishing line
    * it is the image of the plane at infinity
    * all vanishing points lie on the vanishing line
- image of the absolute conic (IAC)
    * it is useful for camera calibration!




# todo
adaboost, graphical models, SVM
face recognition (pattern matching)

boosting gbdt xgb lgb, auc area, roc curve, light gbm
random forest, xgboost

crf (conditional random field)

word2vec, lstm

[CS230: Deep Learning][45]

[Single View Metrology][59] and
this [assignment][60] and
refer to [this one][61]


[61]: http://www.robots.ox.ac.uk/~vgg/projects/SingleView/otherlinks.html
[60]: http://www.cs.cmu.edu/~ph/869/src/asst3/asst3.html
[59]: http://www.cs.cmu.edu/~ph/869/papers/Criminisi99.pdf
[58]: http://www.vision.is.tohoku.ac.jp/files/2714/9360/0441/2-camera_calibration.pdf
[57]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node46.html
[56]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node45.html
[55]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node43.html
[54]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node44.html
[52]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node43.html
[51]: http://www.cse.iitd.ernet.in/~suban/vision/geometry/node40.html
[50]: https://www.springer.com/cda/content/document/cda_downloaddocument/9780857290458-c2.pdf?SGWID=0-0-45-998549-p174031075
[49]: http://www.cse.iitd.ernet.in/~suban/vision/affine/node6.html
[48]: http://www.cse.iitd.ernet.in/~suban/vision/affine/node5.html
[47]: http://www.cse.iitd.ernet.in/~suban/vision/affine/node3.html
[46]: http://www.cse.iitd.ernet.in/~suban/vision/affine/node2.html
[45]: https://web.stanford.edu/class/cs230/
[43]: https://archive.ics.uci.edu/ml/datasets/iris
[42]: http://rcs.chph.ras.ru/Tutorials/classification/Fisher.pdf
[41]: https://link.springer.com/content/pdf/10.1007/BF00994018.pdf
[40]: http://web.stanford.edu/class/cs224n/
[39]: http://neuralnetworksanddeeplearning.com/index.html
[38]: http://cs231n.stanford.edu/
[37]: http://ml.informatik.uni-freiburg.de/former/teaching/ws1617/dl.html
[4]: http://www.seanborman.com/publications/EM_algorithm.pdf
[3]: http://www.eng.auburn.edu/~troppel/courses/7970%202015A%20AdvMobRob%20sp15/literature/paper%20W%20refs/dempster%20EM%201977.pdf
[2]: https://arxiv.org/pdf/1105.1476.pdf
[1]: http://cs229.stanford.edu/notes/cs229-notes8.pdf
