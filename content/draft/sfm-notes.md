---
title: "Sfm Notes"
date: 2018-11-10T10:55:55+08:00
draft: true
tags: [
    "computer_vision",
    "mvg",
    "sfm",
]
categories: [
    "Development",
]
---

Sfm is not the same as SLAM!

# Descriptors
- Harris corner
    * the formulation of the problem for corner detection
    * gradient, structure tensor, compute the minimum eigen value of the structure tensor, non-maximum suppression
- [FAST][4], ECCV, 2006, Features from Accelerated Segment Test
    * in a 7x7 neighborhood, choose 16 neighbors fitting in a circle sequentially
    * a corner: a 12 sequential neighbors that are brighter or darker than the current pixel
    * it is just a corner detector
    * no orientation information is available
- [BRIEF][2], paper, ECCV2010, Binary Robust Independent Elementary Features
    * it is similar to census transform!!!
    * in BRIEF, the common settings are 128, 256 or 512 neighbors
    * the selection of neighbors is not the same as census transform, it selects
them randomly (i.e., via a gaussian distribution)!
    * Drawbacks: no scale/rotation invariance, sensitive to noise
- [ORB][3], paper, ICCV2011
    * Oriented FAST and Rotated BRIEF
    * improvement for FAST: (1) use harris corner detector for filtering;
(2) repeat the process across pyramids (similar to SIFT); (3) add orientation:
first compute the intensity centroid in the patch, then connect the centroid and the corner
and take the angle of the vector as the orientation
    * that is why it is called Oriented FAST
    * improvement for ORB: (2) only compute location detected by FAST
(3) before doing the binary test, rotate the point of the neighbor by $\theta$
computed from oriented FAST; it uses a lookup table to get the rotated location.
this is called *Steered BRIEF*

- SIFT
    * 128 floats, very slow, not suitable for SLAM, but it is used in Sfm
- SURF,
    * 64 floats, faster than SIFT,
    * opencv implementation
- DAISY
- GIST

# References

- [CVPR 2017 Tutorial on Large-scale 3D Modeling from Crowdsourced Data.][1]


[4]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.3991&rep=rep1&type=pdf
[3]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.4395&rep=rep1&type=pdf
[2]: https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
[1]: https://demuc.de/tutorials/cvpr2017/
