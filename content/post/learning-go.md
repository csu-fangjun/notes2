---
title: "Learn Go Programming"
date: 2018-10-16T17:16:33+08:00
tags: [
    "go",
    "golang",
]
categories: [
    "Development",
    "golang",
]
draft: false
---

## TODO

0. How to write `hello world` in go
1. Structure of a project
    - packages
      * package initialization function: `func init() {...}` which is called automatically when it is imported by others (<font color="red">how many times is it called?</font>)
2. Visibility of variables and functions  
    - upper case (exported) vs. lower case
    - package level, function level
3. How to define:
    - variables
      * the special variable: `_`
      * the initialization value for implicit initialization, i.e., default values
      * primitive data types
      * short variable declaration, i.e., `:=`
      * tuple assignment like python
      * `new()`
      * `make()`
    - constants
      * enums (i.e., with `iota`)
    - functions
      * it can be defined in any order in any file in the same package
      * no need for forward declaration
      * the same goes for variables
    - structure
      * and its methods
    - maps
    - interfaces
    - arrays and slices
    - pointers
      * how to dereference
      * pointer arithmetic via `unsafe.Pointer()`
      * the nullptr is `nil`
      * a pointer to a local variable can be returned which is very different from C/C++ because of the garbage collector in Go
4. Source code of predefined modules
    - fmt
    - bytes
    - flag
    - fmt
    - io
    - log [^8]
        * `log.Fatalf()`
        * `log.Printf()`
    - math
    - path
    - strconv
    - strings
    - time
    - errors
    - encoding, encoding/json
    - regexp
5. Naming conventions
    - camel case and visibility
6. How to print variables
    - `fmt.Printf()`
      * `\n`, `%T`, `%v`, `%d`, `%s`, `%c`, `%b`, `%o`, `%x`
    - `fmt.Println()`
    - `fmt.Print()`
    - `fmt.Fprintf()`
    - `fmt.Sprintf()`
    - `fmt.Errorf()`

## Numerical Data Types
### Integers
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `int`, the compiler decides whether it is 32 or 64 bits
- `uint`, the compiler decides whether it is 32 or 64 bits
- `rune`, <font color="red">**r**</font>presentation of <font color="red">**un**</font>icod<font color="red">**e**</font> point, which is `int32`
- `uintptr`, the compiler decides its width which is at least able to store a pointer

The following code is valid

```go
var a rune
var b int32 = a
```

But the following are invalid and result in compile time error

```go
var a rune
var b int = a       /* error, assign rune to int */
var c int64 = a     /* error, assign rune to int64 */
var d int32 = b     /* error, assign int to int32 */
var e int64 = a     /* error, assign int to int64 */
```

#### Limits

Similar to `std::numeric_limits<xxx>::max()` in C++, there are predefined
limits for integers in the `math` module: [^2]

```go
fmt.Println(math.MinInt8)  /* -128 */
fmt.Println(math.MaxInt8)  /* 127 */
fmt.Println(math.MinInt16) /* -32768 */
fmt.Println(math.MaxInt16) /*  32767 */
fmt.Println(math.MinInt32) /* -2147483648 */
fmt.Println(math.MaxInt32) /*  2147483647 */
fmt.Println(math.MinInt64) /* -9223372036854775808*/
fmt.Println(math.MaxInt64) /*  9223372036854775807*/
```

### Floating Point Numbers

- `float32`
- `float64`
- Unlike C/C++, there is **no** `float` or `double` type in Go
- Notations like `1.2f` is invalid in Go!

```go
var a float32
var b float64
a = 1.2
b = 1.2
c := 1.2
d := float32(1.2)
fmt.Printf("%T\n", a) /* float32 */
fmt.Printf("%T\n", b) /* float64 */
fmt.Printf("%T\n", c) /* float64 */
fmt.Printf("%T\n", d) /* float32 */
```

The above code shows that the default type for floating point numbers
is `float64`.

#### NaN and Inf

Go uses `float64` to represent `NaN` and `Inf`[^7].

A NaN has the following bit representation

```
01111111 1111xxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
```
with at least one nonzero `x`.

Go sets the first `x` to 1 which means quiet NaN [^3]. The last `x` is
also set to 1 and remaining `x`s are set to 0. Thus the quiet NaN
in Go is represented by `0x7f f8 00 00 00 00 00 01`[^4].

A positive `Inf` sets all `x` to 0 and a negative `Inf` sets
the sign bit of the positive `Inf` to 1.

- positive `Inf` in Go: `0x7f f0 00 00 00 00 00 00` [^5]
- negative `Inf` in Go: `0xff f0 00 00 00 00 00 00` [^6]

```go
fmt.Println(math.NaN())                              /* NaN */
fmt.Printf("%#0x\n", math.Float64bits(math.NaN()))   /* 0x7ff8000000000001 */
fmt.Println(math.IsNaN(math.NaN()))                  /* true */
fmt.Println(math.Inf(1))                             /* +Inf */
fmt.Println(math.Inf(-1))                            /* -Inf */
fmt.Println(math.IsInf(math.Inf(1), 1))              /* true */
fmt.Println(math.IsInf(math.Inf(-1), -1))            /* true */
fmt.Printf("%#0x\n", math.Float64bits(math.Inf(1)))  /* 0x7ff0000000000000 */
fmt.Printf("%#0x\n", math.Float64bits(math.Inf(-1))) /* 0xfff0000000000000 */

fmt.Println(math.NaN() != math.NaN())   /* true */
fmt.Println(math.Inf(1) == math.Inf(1)) /* false */

a := 0.
fmt.Println(0 / a) /* NaN */

fmt.Println(1 / a)  /* +Inf */
fmt.Println(-1 / a) /* -Inf */

fmt.Println(math.Sqrt(-1)) /* NaN */
```

Note that to build a `Inf` we have to specify whether we're wanting
a positive `Inf` or a negative `Inf` and the sign is passed as a number.
Only the sign of the number matters.

## bool

Note that `if 1 {}` is invalid in Go!

```go
var a bool
a = true
b := 1 > 2
fmt.Printf("%T %[1]v\n", a) /* bool true */
fmt.Printf("%T %[1]v\n", b) /* bool false */
fmt.Printf("%t\n", b) /* false */
```

Note that we cannot use `%d` to print a `bool`! `%t` can be used instead

The meaning of `&&` and `||` is the same as C/C++,
i.e., short-circuit evaluation.

## string
`string` in Go has exactly the same memory layout with `[]byte` but

- we cannot use `&s[1]` to get the address of its element
- there is no `cap()` method and no `cap` field in the header for a `string`
- `s[0] = 'a'` results in compile time error since a string is readonly.

We can use `len()` to get the number of bytes in a `string`.
Number of bytes in a `string` does not necessarily equal to
the number of characters (i.e., rune) in a `string`. For example, `"中"`
is one character (or one rune) but it occupies 3 bytes: `0xe4`, `0xb8`, `0xad`
which are its utf8 encoding. `len("中")` returns 3.

```go
s := "123"
fmt.Printf("%T\n", s) /* string */

// ps: address of s, which is the address of the header
ps := (*uint64)(unsafe.Pointer(&s))

// parr: address of the array
parr := (*uint8)(unsafe.Pointer(uintptr(*ps)))
parr1 := (*uint8)(unsafe.Pointer(uintptr(unsafe.Pointer(parr)) + 1))
parr2 := (*uint8)(unsafe.Pointer(uintptr(unsafe.Pointer(parr)) + 2))

fmt.Printf("%x %x %x\n", *parr, *parr1, *parr2) /* 31 32 33 */

// *parr2 = '4' /* runtime error! */

plen := (*int)(unsafe.Pointer((uintptr(unsafe.Pointer(ps)) + 8)))
fmt.Println("len:", *plen) /* len: 3 */

// string has no capacity field!
// pcap := (*int)(unsafe.Pointer((uintptr(unsafe.Pointer(ps)) + 16)))
// fmt.Println("cap:", *pcap) /* cap: 0 */
```

### Raw Literal String

Use \` \` to build a literal string.

```go
s1 := `this is a newline \n
a second line
`

fmt.Printf("%T\n", s1) /* string */
fmt.Println(s1)
/*
    this is a newline \n
            a second line
*/

s2 := "this is a newline \\n\na second line"
fmt.Println(s2)
/*
    this is a newline \n
    a second line
*/

/* Note that for ``, \n do not need to be skipped.
  Moreover, a "second a line" is indented for ``.
*/
```

### byte slice to string

```go
	b := []byte{'a', 'b', 'c'}

	var s string
	s = *(*string)(unsafe.Pointer(&b))

	fmt.Println(s) /* abc*/

	b[0] = 'd'
	fmt.Println(s) /* dbc*/

	b = b[:2]      /*a new header is assigned to b, but s still points to the original header*/
	fmt.Println(s) /* dbc*/
```

Note that `s` and `b` shares the same header. The capacity field
in the header of `b` is not used by `s`.

## Array
```go
	var a [3]int
	fmt.Println(a)              /* [0 0 0 ] */
	fmt.Printf("%v\n", a)       /* [0 0 0] */
	fmt.Println(len(a), cap(a)) /* 3 3 */

	b := [3]int{1, 2, 3}
	fmt.Println(b)              /* [1 2 3] */
	fmt.Println(len(b), cap(b)) /* 3 3*/

	c := [...]int{1, 2}
	fmt.Printf("%T\n", c)       /* [2]int */
	fmt.Println(c)              /* [1 2] */
	fmt.Println(len(c), cap(c)) /* 2 2 */

	c[0] = 10
	fmt.Println(c) /* [10 2] */

	// c[3] = 20    /* compile time error */

	for i, v := range c {
		fmt.Printf("c[%d] = %d, ", i, v) /* c[0] = 10, c[1]=2, */
		v = 0                            /*v is a copy, c is not changed!*/
	}
	fmt.Println()
	for i := range c {
		fmt.Printf("c[%d] = %d, ", i, c[i]) /*c[0] = 10, c[1] = 2, */
	}

	fmt.Println()
	for _, v := range c {
		fmt.Printf("%d, ", v) /*10, 2, */
	}

	fmt.Println()
	fmt.Printf("&c=%p\n&c[0]=%p\n&c[1]=%p\n", &c, &c[0], &c[1])
	/*
		&c   =0xc0000180e0
		&c[0]=0xc0000180e0
		&c[1]=0xc0000180e8
	*/

	d := [2]int{
		1: 200,
		0: 100,
	}
	fmt.Println(d) /* [100 200] */

	e := [...]int{
		3: 300,
	}
	fmt.Printf("%T\n", e) /* [4]int */
	fmt.Println(e)        /* [0 0 0 300] */

	f := [2]int{10, 20}
	g := [2]int{100, 200}
	f = g          /* copy */
	f[0] = 0       /* g is not changed */
	fmt.Println(f) /* [0 200] */
	fmt.Println(g) /* [100 200]*/

	f = g
	fmt.Println(f == g) /* true */
	fmt.Println(f != g) /* false */
	fmt.Printf("&f=%p\n&g=%p\n", &f, &g)
	/*
		&f=0xc000018100
		&g=0xc000018110
	*/
```

- how to print an array
- how to get the length of an array
- how to define an array variable
    * how to let the compiler determine the array length
    * how to initialize an array at the time of definition
- how to access the element of an array
- how to iterate an array
- is an array stored continuously in memory
- unlike C/C++, the name of an array in Go is not a constant
    * `arr1 = arr2` is totally valid and has copy semantics
    * avoid passing an array to a function in Go
    * avoid returning an array from a function
    * pass a pointer or a slice of an array instead!
- arrays of the same type can be compared using `==`, `!=`
    * but comparison with `<=`, `<` etc. results in compile time error!

## Slice

```go
	var a []int
	fmt.Printf("%T\n", a) /* []int */
	fmt.Println(a == nil) /* true */
	fmt.Println(len(a))   /* 0 */
	fmt.Println(cap(a))   /* 0 */

	var b []int = []int{}
	fmt.Printf("%T\n", b) /* []int */
	fmt.Println(b == nil) /* false */
	fmt.Println(len(b))   /* 0 */
	fmt.Println(cap(b))   /* 0 */

	fmt.Println(a, b) /*[] []*/

	d := [3]int{10, 20, 30}
	a = d[1:]

	pa := (*uint64)(unsafe.Pointer(uintptr(unsafe.Pointer(&a))))
	/* pd[0] is invalid in Go ! */
	fmt.Printf("&d=%p\n&d[1]=%p\npa[0]=%#x\n", &d, &d[1], *pa)
	pa1 := (*int)(unsafe.Pointer(uintptr(unsafe.Pointer(pa)) + 8))
	pa2 := (*int)(unsafe.Pointer(uintptr(unsafe.Pointer(pa)) + 16))
	fmt.Printf("pa[1]=%d\npa[2]=%d\n", *pa1, *pa2)
	/*
		&d   =0xc0000161a0
		&d[1]=0xc0000161a8
		pa[0]=0xc0000161a8
		pa[1]=2
		pa[2]=2
	*/
	fmt.Println(len(a)) /* 2 */
	*pa1 = 3
	fmt.Println(len(a)) /* 3 */
	*pa1 = 2

	fmt.Println(cap(a)) /* 2 */
	*pa2 = 1
	fmt.Println(cap(a)) /* 1 */
	*pa2 = 2

	fmt.Println(a, d) /* [20 30] [10 20 30] */
	*(*int)(unsafe.Pointer(uintptr(*pa))) = -1
	fmt.Println(a, d) /* [-1 30] [10 -1 30] */
```

- How to define a slice
    * when is a slice `nil`
- How does the slice header looks like
    * `[0]`: the address of the pointed element in the array.
    * `[1]`: `int`, length of the slice, returned by `len()`
    * `[2]`: `int`, capacity of the slice, returned by `cap()`

### Slice Manipulations

```go
	a := []int{}
	fmt.Println(a == nil)          /*false*/
	fmt.Printf("%T\n", a)          /*[]int*/
	fmt.Println(len(a), cap(a), a) /*0 0 []*/

	a = make([]int, 0)
	fmt.Println(a == nil)          /*false*/
	fmt.Println(len(a), cap(a), a) /*0 0 []*/

	// a = new([]int) /* compile time error*/

	b := new([]int)
	fmt.Printf("%T\n", b) /* *[]int */

	a = make([]int, 2)
	fmt.Println(len(a), cap(a), a) /*2 2 [0 0]*/
	a = make([]int, 2, 3)
	fmt.Println(len(a), cap(a), a) /*2 3 [0 0]*/

	a = append(a, 10)
	fmt.Println(len(a), cap(a), a) /*3 3 [0 0 10]*/

	a = append(a, 20)
	fmt.Println(len(a), cap(a), a) /*4 6 [0 0 10 20]*/

	a = append(a, 30, 40)
	fmt.Println(len(a), cap(a), a) /*6 6 [0 0 10 20 30 40]*/

	a = append(a, 50)
	fmt.Println(len(a), cap(a), a) /*7 12 [0 0 10 20 30 40 50]*/

	a = append(a, a...)
	fmt.Println(len(a), cap(a)) /*14 24*/

	a = append(a, a...)
	fmt.Println(len(a), cap(a)) /*28 48*/

	a = append(a, make([]int, 69)...)
	fmt.Println(len(a), cap(a)) /*97 112*/
```
- how to append a slice
    * how to append variable elements at once
    * how to unpack

## Map

Maps in Go are based on hash tables.

- how to define a map
    * key, value
    * what are the constraints of an key
- Note that access a non-existent key does not create the key automatically!
- how to access a key
- how to delete a key
- how to iterate a map

```go
	var b map[int]string
	fmt.Println(b == nil) /* true */

	a := map[int]string{}
	fmt.Println(a == nil) /* false */
	fmt.Println(len(a))   /* 0 */

	if value, ok := a[0]; !ok {
		fmt.Println("a[0] does not exist!") /*choose this*/
	} else {
		fmt.Println("a[0] is", value)
	}
	/*the above a[0] does not create a new element!*/
	fmt.Println(len(a)) /* 0 */

	a[0] += "hello" /*even a[0] does not exist, += can be used on it! */
	if value, ok := a[0]; !ok {
		fmt.Println("a[0] does not exist!")
	} else {
		fmt.Println("a[0] is", value) /* a[0] is hello */
	}
	fmt.Println(len(a)) /* 1 */

	a[-1] = "world"
	fmt.Println(len(a)) /* 2 */
	for key, value := range a {
		fmt.Printf("key: %d, value: %s\n", key, value)
	}
	/*
		key: 0, value: hello
		key: -1, value: world
	*/

	for key := range a {
		fmt.Println(key)
	}
	/*
		0
		-1
	*/
	delete(a, -1)       /*remove key -1*/
	fmt.Println(len(a)) /* 1 */

	delete(a, 100)      /* remove a key that does not exist is valid! */
	fmt.Println(len(a)) /* 1 */

	fmt.Println(a[1])   /* print nothing. */
	fmt.Println(len(a)) /* 1 */
	fmt.Println(a)      /*map[0:hello]*/
	// fmt.Println(&a[0])  /*compile time error: cannot take the address of a[0]*/

	m := map[string]int{
		"hello": 0,
		"world": 1,
	}
	fmt.Println(m) /*map[hello:0, world:1]*/

	d := make(map[int]int)
	fmt.Println(d == nil) /* false */
	fmt.Println(len(d))   /* 0 */
```

## Struct

- both `(*p).field` and `p.field` are valid when `p` is a pointer in Go!
- there is no `p->field` in Go!
- a `struct` can be defined either inside a function or outside of a function
    * i.e., `type xxx struct` can appear either in a function or outside of it
- the memory layout of a `struct` in Go is similar to C/C++
    * that is, there might be **padding** added by the compiler between fields
    * a `struct` is stored continuously in the memory
- a `struct` can be exported or not exported and at the same time
    * part of its fields can be exported
    * part of its fields can be not exported
    * i.e., even when a `struct` is exported, we cannot access all of its element
- Like `array`s, a `struct` is passed by copy to a function!
- <font color="red">**anonymous fields**</font>

```go
	type Student struct {
		grade int16
		age   int8
		id    int32
	}
	s1 := Student{1, 2, 3}
	fmt.Println(s1) /* 1 2 3 */
	s2 := Student{
		id:    1,
		age:   10,
		grade: 100,
	}
	fmt.Println(s2) /* 10 10 1*/

	var p *Student
	p = &s2
	p.id = 2
	fmt.Println(s2) /* 10 10 2*/
	// p->id = 3	/*compile time error*/
	(*p).id = 3
	fmt.Println(s2) /* 10 10 3*/

	fmt.Printf("&s2=%p\n", &s2)             /*&s2=0xc0000180f0*/
	fmt.Printf("&s2.grade=%p\n", &s2.grade) /*&s2.grade=0xc0000180f0*/
	fmt.Printf("&s2.age=%p\n", &s2.age)     /*&s2.age=0xc0000180f2*/
	fmt.Printf("&s2.id=%p\n", &s2.id)       /*&s2.id=0xc0000180f4*/
	/*Note that there is one byte padding after s2.grade*/
```

## Functions

- How to declare a function
    * arguments (none, single, multiple, or variadic)
    * return values (none, single, or multiple)
    * the order of declaration/definition does not matter; it can even be at different files as long as they are at the same package

- The function signature in Go contains the return types which is different from
C++!!!

- There are no default arguments and no keyword arguments in Go
- Like C/C++, function parameters are pass by value. Unlike C++, there is no
reference type in Go. Use pointers as function arguments
or use slices when necessary.

```go
func sayHelloWorld(str string) {
	fmt.Println(str + " hello " + sayWorld())
}

func main() {
	sayHelloWorld("--") /* -- hello world */
}

func sayWorld() string {
	return "world"
}
```

### Named Return values

If return values are given a name, they are initialized to its default value.
Note that even if the return values are named, we still have to use `return`
inside the function.

```go
func named1() (a int) {
	return a /* a is initialized to its default value 0*/
}

func named2() (b int) {
	b = 3
	return b
}

func named3() (a int, b string) {
	a = 10
	b = "hello"
	return a, b /* return multiple values! */
}

func main() {
	a := named1()
	fmt.Println(a) /* 0 */

	b := named2() /* 3 */
	fmt.Println(b)

	c, d := named3()
	fmt.Println(c, d) /* 10 hello */
}
```

### Variables of Function types

- How to define a variable of a function type
    * use `var`
    * use short declaration

```go
func helloWorld() string {
	return "hello world"
}

func halloWelt() {
	fmt.Println("Hallo Welt!")
}

func main() {
	var f func() string
	f = helloWorld
	fmt.Println(f()) /* hello world */

	g := halloWelt
	g() /* Hallo Welt! */
}
```

### Static Variables

Use closure to simulate static variables in Go.

The following code captures `x` by reference.
The `x` in `f` and `f2` are two different copies.

```go
func f_() func() int {
	var x int
	return func() int {
		x++
		return x
	}
}

func main() {
	f := f_()
	fmt.Println(f()) /* 1 */
	fmt.Println(f()) /* 2 */
	fmt.Println(f()) /* 3 */
	fmt.Println(f()) /* 4 */

	f2 := f_()
	fmt.Println(f2()) /* 1 */
}
```

### Variadic Functions

It is convention in Go that a variadic function name ends with `f`;
for example, `fmt.Printf()`.

```go
func f(arr ...int) {
	fmt.Printf("%T %d %d %v\n", arr, len(arr), cap(arr), arr)
}

func main() {
	a := []int{1, 2, 3}

	f()     /* []int 0 0 [] */
	f(a...) /* []int 3 3 [1 2 3] */
	f(0)    /* []int 1 1 [0] */
	f(0, 2) /* []int 2 2 [0 2] */
}
```

### defer

`defer xxx()()`: `xxx()` is executed when `defer` is encountered.
The last `()` is called at the end of the function.


```go
func log(x int) (result int) {
	fmt.Println("enter")
	defer func() {
		result += 10
	}()

	result += x
	return result
}

func main() {
	fmt.Println(log(1))
	/*
		enter
		11
	*/
}
```


The following code shows that `defer` is called in "first in last out" manner.
`x` is evaluated when `defer` is encountered.

```go
func main() {
	x := 1
	defer fmt.Println(x)

	x += 1
	defer fmt.Println(x)

	x += 1
	defer fmt.Println(x)

	/*
		3
		2
		1
	*/
}
```

### panic
```go
func f3() {
	panic("some message")
}

func f2() {
	f3()
}

func f1() {
	f2()
}
func main() {
	f1()
}
```

prints the following calling stack information

```
panic: some message

goroutine 1 [running]:
main.f3(...)
	/xxx/go/src/hello/hello.go:6
main.f2(...)
	/xxx/go/src/hello/hello.go:10
main.f1(...)
	/xxx/go/src/hello/hello.go:14
main.main()
	/xxx/go/src/hello/hello.go:17 +0x39
exit status 2
```

## Methods

- What is a receiver in a method
    * value receiver
    * pointer receiver

For a value receiver, we can call it with a pointer or with a value; it's
impossible to change the fields of the receiver even if we use a pointer when
it is a value receiver.

For a pointer receiver, we can call it either with a value
or a pointer; we can change the field of the struct even if we use
a value when it is a pointer receiver.

```go
type S struct {
	i int
}

func (s S) value() {
	s.i += 1
}

func (s *S) pointer() {
	s.i += 1
}

func main() {
	s := S{i: 1}
	fmt.Println(s) /*{1}*/

	s.value()
	fmt.Println(s) /*{1}*/

	s.pointer() /* s.i is changed!*/
	fmt.Println(s) /*{2}*/

	p := new(S)
	p.i = 1
	fmt.Println(p) /*&{1}*/

	p.value() /* p.i is not changed! */
	fmt.Println(p) /*&{1}*/

	p.pointer() /* p.i is changed! */
	fmt.Println(p) /*&{2}*/
}
```

### struct embedding

```go
type Point2 struct {
	x int
	y int
}

type Point3 struct {
	Point2
	z int
}

func main() {
	// p := Point3{1, 2, 3}	/* compile error */

	var p Point3
	p.x = 1
	p.y = 2
	p.z = 3
	fmt.Println(p.x) /* 1 */
	fmt.Println(p.y) /* 2 */
	fmt.Println(p.z) /* 3 */
}
```

```go
	p := Point3{
		Point2{1, 2},
		3,
	}
	fmt.Println(p)        /* {{1 2} 3} */
	fmt.Println(p.Point2) /* {1 2} */
```

### methods for builtin types

```go
type myInt int

func (i *myInt) inc(a int) {
	*i = *i + myInt(a)
}

func main() {
	var i myInt
	i = 2
	i.inc(3)
	fmt.Println(i) /* 5 */
}
```

## Interface

It is similar to the abstract class concept in C++. We must implement all
functions inside the interface; otherwise it is still an abstract class.



## fmt.Printf
Example code and its output:
```go
fmt.Printf("%v\n", []int{3, 4})                           /* 3 4 */
fmt.Printf("%T\n", 10)                                    /* int */
fmt.Printf("%x %#x %3x %#4x %#04X\n", 10, 10, 10, 10, 10) /* a 0x0a     a  0xa 0X000A */
fmt.Printf("%x %#[1]x %3[1]x %#4[1]x %#04[1]X\n", 10)     /* a 0x0a     a  0xa 0X000A */
fmt.Printf("%d %[2]d %[1]d\n", 1, 2)                      /* 1 2 1 */
fmt.Printf("%c %[1]q\n", 'a')                             /* a 'a'*/
fmt.Printf("%c %[1]q %#[1]x %+[1]q %[1]U\n", '中')         /* 中 '中' 0x4e2d '\u4e2d' U+4E2D */
fmt.Printf("%#U\n", '中')                                  /* U+4E2D '中' */
fmt.Printf("%s %[1]v %[1]q\n", "中国")                      /* 中国 中国 "中国" */
fmt.Printf("%x\n", "中123")                                /* e4b8ad313233 */
fmt.Printf("% x\n", "中123")                               /* e4 b8 ad 31 32 33 */
fmt.Printf("%q\n", "中123")                                /* "中123" */
fmt.Printf("%+q\n", "中123")                               /* \u4e2d123*/
```
 - `%v` to print the value of the argument, useful for `array`, `slice`, etc.
 - `%T` to print the type of the argument
 - `#` is to print `0x` for `%x` and `0` for `%o`
 - `[1]` is the first parameter in the parameter list. It counts from **1** instead of 0
 - `[2]` is the second parameter in the parameter list
 - difference between `%x` and `%X`:
     * `%x` prints lower cases, i.e, 0x, abcdef
     * `%X` prints upper cases, i.e, 0X, ABCDEF
 - `%q` to add `''` to output characters and add `""` to strings
 - `%+q` to print the unicode code point for strings and characters
 - `%x` to print unicode for characters but utf8 for strings
 - `%U` can be used only for characters to print its unicode point
 - `%#U` to print both the unicode point and the corresponding character
 - Note that `%u` is undefined in Go! (It is for unsigned in C/C++)
 - both a character and a `rune` can be printed with `%c` or `%q`
 - the unicode code point [^1] for `中` is `U+4E2D`; use `%x` to view its unicode code point
 - the utf8 encoding [^1] for `中` is `e4 b8 ad`
 - refer to "Strings, bytes, runes and characters in Go" (https://blog.golang.org/strings) for string representations in Go. The utf8 code point of a string is stored in the memory.

## Common Errors

### Name Equivalence instead of Structural Equivalence

```go
type Int int
var a Int
var b int = a
```

Since `a` is of type `Int` and `b` is of type `int`, the above assignment
results in a compile time error:

> cannot use a (type Int) as type int in assignment

<font color="red">**Note**:</font> Both `a < b` and `a*b` are **invalid**!

To fix the error, we can use explicit conversion from `Int` to `int`:

```go
type Int int
var a Int
var b int = int(a)  /* now no error occurs!
```

# Packages

## Packages with the same name
```go
import (
	"a/foo"
	bfoo "b/foo"
)
```
- `foo.someFunc()` refers to `someFunc()` in `a/foo`
- `bfoo.bar()` refers to `bar()` in `b/foo`

## Package Initializer
```go
func init() {
    /*bla bla...*/
}
```
The name has to be `init()`, which is executed when the package is imported
by another package.

## Download packages
```sh
go get github.com/xxx/yyy
```

# Test

Create a file `hello.go`:

```go
package hello

func add(a, b int) int {
	return a + b
}
```

Create another file `add_test.go`:

```go
package hello

import "testing"

func TestMyAdd(t *testing.T) {
	if add(1, 2) == 3 {
		t.Error("error msg1")
		t.Error("error msg2")
	}
	t.Error("error agian")
}

func TestAdd(t *testing.T) {
	if true {
		t.Error("it is an error!")
	}
	t.Error("error2")
}
```

Run the command `go test`, which outputs

```
--- FAIL: TestMyAdd (0.00s)
    add_test.go:7: error msg1
    add_test.go:8: error msg2
    add_test.go:10: error agian
--- FAIL: TestAdd (0.00s)
    add_test.go:15: it is an error!
    add_test.go:17: error2
FAIL
exit status 1
FAIL	hello	0.005s
```

# The unsafe Package

```go

```


[^8]: https://github.com/golang/go/tree/master/src/log
[^7]: https://github.com/golang/go/blob/master/src/math/bits.go
[^6]: https://github.com/golang/go/blob/master/src/math/bits.go#L10
[^5]: https://github.com/golang/go/blob/master/src/math/bits.go#L9
[^4]: https://github.com/golang/go/blob/master/src/math/bits.go#L8
[^3]: https://en.wikipedia.org/wiki/NaN#Encoding
[^2]: Refer to https://github.com/golang/go/blob/master/src/math/const.go#L39
[^1]: Refer to http://www.ltg.ed.ac.uk/~richard/utf-8.cgi?input=%E4%B8%AD&mode=char for conversion between unicode utf8.
