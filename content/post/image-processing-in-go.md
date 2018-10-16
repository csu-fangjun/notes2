---
title: "Image Processing in Go"
date: 2018-10-29T20:06:40+08:00
draft: false
tags: [
    "go",
    "golang",
]
categories: [
    "Development",
    "golang",
]
---

The origin is the same with OpenCV and the memory layout is similar to
OpenCV except that Go uses `RGBA` order and it might not be continuous,
i.e., there will be some paddings at the end of each row.

## Setup

```sh
go get -u github.com/disintegration/imaging
```

## The image/color package

```go
type Color interface {
	RGBA() (r, g, b, a uint32)
}
```
Note that the `Color` interface has a function `RGBA()` returning
four values, each of which is of type `uint32`. Only the lower 16 bits
are valid. That is, color values in Go range from 0 to 65535.

In OpenCV, a red component has only 8-bit representing values from 0 to 255;
while Go requires a red color to have values from 0 to 65535.

65535/255=257, which is `(1<<8) + 1`. For any color `r`, we have to perform
the following conversion `r = r*257` to make it in the range from 0 to 65535.

```go
type RGBA struct {
	R, G, B, A uint8
}

func (c RGBA) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R)
	r |= r << 8
	g = uint32(c.G)
	g |= g << 8
	b = uint32(c.B)
	b |= b << 8
	a = uint32(c.A)
	a |= a << 8
	return
}
```

Go provides a `RGBA` struct which implements the `Color` interface. Pay attention
to the difference between the struct `RGBA` and the function `RGBA()`!
Because the struct fields of `RGBA` are of type `uint8`, we have to scale it
to 65535 via `r |= r << 8`, which is equivalent to `r =  r + 256*r` or
`r = r*257`.

# Point
In `image/geom.go`:
```go
// A Point is an X, Y coordinate pair. The axes increase right and down.
type Point struct {
	X, Y int
}

// String returns a string representation of p like "(3,4)".
func (p Point) String() string {
	return "(" + strconv.Itoa(p.X) + "," + strconv.Itoa(p.Y) + ")"
}
```

Pay attention to `strconv.Itoa()` for converting an integer to a string.






## References
- [The Go image package][1]
- [The Go image/draw package][2]


[2]: https://blog.golang.org/go-imagedraw-package
[1]: https://blog.golang.org/go-image-package
