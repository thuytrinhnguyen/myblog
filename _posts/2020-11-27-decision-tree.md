---
layout: post
title:  "Decision Tree"
date:   2020-11-27 15:44
categories: unsupervised-learning
tags: decision-tree classification
---

> Intro

<!--more-->

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}


## Purpose

* Ask questions

* Classify the data based on the answer

Example

Yes/No

Rank (score 1, 2)

- Leaves can be numeric or categorical

- Can have different questions on each side


## Notation

Root Node

Internal Nodes: 2 types of arrow

Leaf Nodes: Final node (classification)

## How to build a decision tree

### 1. Choose the root node

- How well a variable predict

- Measure Impurity: Which one separate the classes the best (less mixed)

Calculate GINI... for all variables

Choose the variable with lowest GINI (impurity) to be the root node

[StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk&t=1s)

If the impurity does not reduce after another step => Leaf node (13:42)

__GINI__

$$
G = 1 - \sum_{i} p_{c_i} ^{2}
$$

__Entropy__

$$
H = - \sum_{i}p(c_i) \times log_{2}p(c_i)
$$




### Deal with numerical data (Dont include everything - Find a breaking point => half the option)

1. Continuous

Sort data

Calculate average weight of every 2 data points = Breaking point

Calculate Gini for all breaking points

Choose one with the lowest Gini


2. Discrete (ranking 1 2 3)

Calculate gini for each rank (not average values)



### 2. 

