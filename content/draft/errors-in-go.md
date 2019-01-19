---
title: "Errors in Go"
date: 2018-10-28T15:12:34+08:00
draft: true
tags: [
    "go",
    "golang",
]
categories: [
    "Development",
    "golang",
]
---

# Introduction
This post contains my notes for error handling in Go.

# The Type: `error`
[`error`][1] is a [built-in interface type][2] in `Go`.

It is defined as follows (you cannot find it in the source code!):

```go
type error interface {
    Error() string
}
```

Even if `error` begins with lower case, it is accessible
by others just like types `int`, `int16`, etc.

# The Package: `errors`

The package [`errors`][3] contains 3 files:

- [errors.go][4]
- [errors_test.go][5]
- [example_test.go][6]

## errors.go

First it declares the package name:

```go
package errors
```

There is no `import` statement. Then it defines a `struct`

```go
type errorString struct {
    s string
}
```

Note that `errorString` begins with a lower case `e`, so it is not exported.
The `strcut` implements the `error` interface:

```go
func (e *errorString) Error() string {
    return e.s
}
```

Note that the receiver is a pointer!

Since `errorString` is not exported, a wrapper is provided

```go
func New(text string) error {
    return &errorString{text}
}
```

`New()` returns a interface `error`. Inside the function a pointer
is returned, which implements the `error` interface!




[6]: https://github.com/golang/go/blob/master/src/errors/errors.go
[5]: https://github.com/golang/go/blob/master/src/errors/errors.go
[4]: https://github.com/golang/go/blob/master/src/errors/errors.go
[3]: https://github.com/golang/go/tree/master/src/errors
[2]: https://blog.golang.org/error-handling-and-go
[1]: https://golang.org/ref/spec#Predeclared_identifiers

