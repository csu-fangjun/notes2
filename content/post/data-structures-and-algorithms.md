---
title: "Data Structures and Algorithms"
date: 2018-10-21T08:27:20+08:00
tags: [
    "data_structures_and_algorithms",
]
categories: [
    "Development",
]
draft: false
---


# Trees
## Binary Search Trees

- BST: Binary Search Tree
- how to build a BST
- operations: search, insert, and delete an item
- complexity of every operation

### Balanced BST

- motivation to balance a binary tree
- also known as AVL trees, AVL stands for two people:
Adelson-Velskii and Landis and was proposed in 1962 in
the paper [An algorithm for the organization of information][1]
- also known as height balanced tree
- advantages: search
- disadvantages: insert and delete
- how to merge and split??

## Splay tree

- also known as self adjusting binary search trees
- motivation, difference from the binary search trees and AVL trees
- proposed in 1985 in the paper [Self-Adjusting Binary Search Trees][4]
- advantages
- disadvantages: i.e., not safe to access in multi-threading (even if it is read only)

## Red-Black trees

- first proposed in 1986 in the paper [A Dichromatic Framework for Balanced Trees][5]

## Dictionary

- Operations: insert, delete, find

# Set and Map in C++ STL

- `#include <set>`: `std::set`, `std::multiset`
- `#include <unordered_set>`, `std::unordered_set`, `std::unordered_multiset`
- `#include <map>`: `std::map`, `std::multimap`
- `#include <unordered_map>`: `std::unordered_map`, `std::unordered_multimap`

All of them supports the following functions:

- `begin()`, `end()`, `size()`, `clear()`,
- `count()`, `find()`
- `insert()`, `erase()`


# TODO
- RB tree
- KMP for substring search
- Hash table

# Books

- [Data structures and algorithm analysis in C++ / Mark Allen Weiss][2], pdf
- [Introduction to Algorithms by Cormen, Leiserson, Rivest, and Stein][3] (CLRS), 3rd, pdf



[5]: http://www.mkurnosov.net/teaching/uploads/DSA/guibas78-red-black-tree.pdf
[4]: http://www.cs.princeton.edu/courses/archive/fall07/cos521/handouts/self-adjusting.pdf
[3]: https://mcdtu.files.wordpress.com/2017/03/introduction-to-algorithms-3rd-edition-sep-2010.pdf
[2]: http://iips.icci.edu.iq/images/exam/DataStructuresAndAlgorithmAnalysisInCpp_2014.pdf
[1]: http://professor.ufabc.edu.br/~jesus.mena/courses/mc3305-2q-2015/AED2-10-avl-paper.pdf




