---
title: "Camera Models"
date: 2018-11-08T12:16:53+08:00
draft: true
tags: [
    "computer_vision",
    "mvg"
]
categories: [
    "Development",
]
---

# Questions

1. What is a pinhole camera model?
2. What is the meaning of $\mathbf{K}[\mathbf{R} \ \mathbf{T}]$:
    * what is $\mathbf{K}$
    * what is $\mathbf{R}$
    * what is $\mathbf{T}$
3. Are $(\mathbf{R}, \mathbf{T})$ the pose of the world frame in the camera frame?
4. Or are $(\mathbf{R}, \mathbf{T})$ the pose of the camera frame in the world frame?
5. If the pose of the world frame in the camera frame is $(\mathbf{R}, \mathbf{T})$
and the camera center in the world frame is $\mathbf{C}$, what is the
relationship among $(\mathbf{R}, \mathbf{T})$ and $\mathbf{C}$?
6. If we have the camera projection matrix $\mathbf{P}$, i.e.,
$\lambda \mathbf{x} = \mathbf{P}\mathbf{X}$, how to find the following entities in the **world** frame:
    * camera center
    * the principle plane
    * the principle axis pointing to the correct direction
    * the principle point (in the image plane)
7. Give a camera matrix $\mathbf{P}$,
    * what is the meaning of its columns (vanishing point of what?)
    * what is the image of the world origin
    * what is the meaning of its first row
    * what is the meaning of its second row
    * what is the meaning of its third row
    * how to compute the depth of a point
    * how to determine whether a point is in front of the camera or not
8. What is a picture of a picture?
9. How to recover $\mathbf{K}$ and $(\mathbf{R}, \mathbf{T})$
from $\mathbf{P}$
10. What is a camera at infinity
    * if its center is at infinity in the world frame
11. what is an affine camera
    * Remember that affine transform keeps the plane at infinity and line at
infinity fixed!
    * an affine camera maps points at infinity to points at infinity

# Epipolar Geometry
1. how to write an API to draw epipolar lines
2. how to do rectification
3. what is view morphing
4. what are the differences between structure from motion
and **affine** structure from motion
    * is the solution unique or defined up to an affine transform ambiguity
    * how many number of equations and unknowns in affine Sfm

# Open Source Implementations
## OpenMVG
Its implements the following [camera models][2]:
- Pinhole without distortion
- Pinhole with 1 radial distortion parameter
- Pinhole with 3 radial distortion parameter
- Pinhole with radial and tangential distortion
- fisheye

### Camera Model Operations

We have to first distinguish two coordinates: the camera coordinate
and the image coordinate (i.e., the pixel coordinate).
In the camera coordinate, we have
$$
\lambda
\begin{bmatrix}
x_u \newline
y_u \newline
1 \newline
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0 & 0 \newline
0 & 1 & 0 & 0 \newline
0 & 0 & 1 & 0 \newline
\end{bmatrix}
\begin{bmatrix}
\mathbf{R} & \mathbf{T} \newline
0 & 1\newline
\end{bmatrix}
\begin{bmatrix}
X \newline
Y \newline
Z \newline
1 \newline
\end{bmatrix}
$$
where $(x_u, y_u)$ is the **undistorted** camera coordinate.
$(0,0)$ is the center in the camera coordinate.

There are several camera distortion models, all of which happen
in the camera coordinate. One distortion model is
$$
\begin{align}
x_d &= (1 + k_1 r_u^2)x_u\newline
y_d &= (1 + k_1 r_u^2)y_u\newline
r_u^2 &= x_u^2 + y_u^2 \newline
\end{align}
$$

The transformation from the camera coordinate to the image (or pixel) 
coordinate is
$$
\begin{bmatrix}
u \newline
v \newline
1 \newline
\end{bmatrix} =
\begin{bmatrix}
f_x & 0 & c_x \newline
0   & f_y & c_y \newline
0 & 0 & 1\newline
\end{bmatrix}
\begin{bmatrix}
x_d \newline
y_d \newline
1 \newline
\end{bmatrix}
$$
Note that the **distorted** camera coordinate is used in the above equation!

One question is that how we can obtain the undistorted camera coordinate $(x_u, y_u)$
from the distorted camera coordinate $(x_d, y_d)$. Take the above distortion model
as an example, we have
$$
\begin{bmatrix}
x_d \newline
y_d \newline
\end{bmatrix} = (1+k_1 r_u^2)
\begin{bmatrix}
x_u \newline
y_u \newline
\end{bmatrix}
$$

$$
\begin{align}
r_d^2 = (1+k_1r_u^2)^2 r_u^2
\end{align}
$$

We are given $(x_d, y_d)$ and $k_1$, if we can get $r_u$, then
$$
\begin{bmatrix}
x_u \newline
y_u \newline
\end{bmatrix} =
\frac{1}{1+k_1 r_u^2}
\begin{bmatrix}
x_d \newline
y_d \newline
\end{bmatrix}
$$

Since $(x_d, y_d)$ is given, we known $r_d$. To get $r_u$, one approach
is bisection using $r_d^2 = (1+k_1 r_u)^2 r_u^2$. The process is as follows:

1. $r_1 = r_2 = r_d$
2. Find $r_1$ such that $(1+k_1 r_1^2)^2 r_1^2 \lt r_d^2$
    * if $(1 + k_1 r_1^2)^2 r_1^2 \gt r_d^2$, then we have to decrease $r_1$
    * to decrease $r_1$, we may use $r_1 = \frac{r_1}{1.05}$
    * repeat decreasing $r_1$ until the above condition is satisfied
3. Find $r_2$ such that $(1+k_1 r_1^2) r_2^2 \gt r_d^2$
    * we can use the same procedure for finding $r_1$ to get $r_2$
    * to increase $r_2$, we can use $r_2 = r_2 \times 1.05$
4. $r = \frac{r_1 + r_2}{2}$
    * if $(1+k_1 r^2)^2 r^2 \gt r_d^2$, then $r = \frac{r_1 + r}{2}$, that is, to decrease $r$
    * if $(1+k_1 r^2)^2 r^2 \lt r_d^2$, then $r = \frac{r + r_2}{2}$, i.e., to increase $r$
    * until the specified iteration number or precision is reached
5. $r_u = r$

To summarize, a camera have to support the following operations:

- conversion from camera coordinate $(x, y)$ to image coordinate $(u, v)$
- conversion from image coordinate $(u, v)$ to camera coordinate $(x, y)$
- add distortion, that is, conversion from $(x_u, y_u)$ to $(x_d, y_d)$
- remove distortion, that is, conversion from $(x_d, y_d)$ to $(x_u, y_u)$

### The pinhole model in OpenMVG
The intrinsic parameters for the [pinhole model][3] is
$$
\begin{bmatrix}
f & 0 & c_x \newline
0 & f & c_y \newline
0 & 0 & 1   \newline
\end{bmatrix}
$$

The limitations are that $f_x=f_y=f$ and there is no skew.

### The Pinhole Model in OpenMVG with 1 Radial Distortion Parameter
The [distortion model][4] is
$$
\begin{align}
x_d &= (1+k_1 r_u^2) x_u \newline
y_d &= (1+k_1 r_u^2) y_u \newline
r_u^2 &= x_u^2 + y_u^2
\end{align}
$$

### The Pinhole Model in OpenMVG with 3 Radial Distortion Parameter
The [distortion model][5] is
$$
\begin{align}
x_d &= (1+k_1 r_u^2 + k_2 r_u^4 + k_3 r_u^6) x_u \newline
y_d &= (1+k_1 r_u^2 + k_2 r_u^4 + k_3 r_u^6) y_u \newline
r_u^2 &= x_u^2 + y_u^2
\end{align}
$$

### The Pinhole Model in OpenMVG with Radial and Tangential Distortion
It is also known as Brown's distortion model.

The [distortion model][6] is
$$
\begin{align}
x_d &= (1+k_1 r_u^2 + k_2 r_u^4 + k_3 r_u^6) x_u  + 2p_1x_u y_u + p_2(r_u^2 + 2x_u^2)\newline
y_d &= (1+k_1 r_u^2 + k_2 r_u^4 + k_3 r_u^6) y_u  + p_1 (r_u^2 + 2y_u^2) + 2p_2x_uy_u\newline
\end{align}
$$

Refer to [here][7] for distortion removal which comes
form the Equation 7 in the paper [Geometric Camera Calibration Using Circular Control Points][8]. Note that it uses `+` instead of `-` for the equation.


# References
- [Elements of Geometric Computer Vision][1]

many functions!!
https://www.peterkovesi.com/matlabfns/#projective



[8]: http://localhost:1313/post/camera_models/#the-pinhole-model-in-openmvg-with-3-radial-distortion-parameter
[7]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/Camera_Pinhole_Brown.hpp#L97
[6]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/Camera_Pinhole_Brown.hpp#L24
[5]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/Camera_Pinhole_Radial.hpp#L282
[4]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/Camera_Pinhole_Radial.hpp#L75
[3]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/PinholeCamera.hpp#L19
[2]: https://github.com/openMVG/openMVG/blob/835f1e585377a55d4143548fa9181438f5d23889/src/openMVG/cameras/Camera_Common.hpp#L39
[1]: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html

