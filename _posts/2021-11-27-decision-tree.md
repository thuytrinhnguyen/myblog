---
layout: post
title:  "Decision Tree"
date:   2021-11-27 15:44
categories: supervised-learning
tags: decision-tree classification regression
---

> This post covers the fundamental concepts in Decision Tree for both regression and classification purposes. From growing a full tree using different splitting criteria to pruning the tree to prevent overfitting, we will go through it all in this blog entry. So let's buckle up!

<!--more-->

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}


## Purpose

Decision tree is a popular technique in supervised learning. The full name of this technique is CART, which stands of Classification-And-Regression-Tree. From its name you can probably guess that Decision tree is capable of both __classification__ and __regression__ problems. In this post, I will walk you through both Regression tree and Classification tree, which share many core ideas.

__So, in general, what does Decision tree do?__

Decision tree builds a set of rules that partition the data in half repeatedly. The ultimate goal is to stratify observations into segments to either classify or predict values of future data points.

{% figure caption: "Fig. 1. Example of a regression tree" class: "width_500"%}
![]({{'/assets/images/decision-tree-reg.png'}})
{% endfigure %}

Fig. 1. demonstrates a regression tree to predict house prices where each leaf node (i.e. the last node on each branch) is the __average__ price of the houses matching the previous conditions. For example, the mean price of all houses that (1) locate in "Inner city" area and (2) have no more than 2 bedrooms, is $400K. Using this regression tree, if a future data points falls into this group, we can predict its price to be $400K.   

{% figure caption: "Fig. 2. Example of a classification tree" class: "width_500"%}
![]({{'/assets/images/decision-tree-classification.png'}})
{% endfigure %}

Classification trees are slightly different as their leaves are categories instead of numerical values. Fig. 2. shows that although the leaf nodes in classification trees cannot be numerical, we can still include numerical predictors in our tree as internal nodes (e.g. Price < $700K). We will discuss the details of how to work numerical independent variables in decision trees in later section. For now, looking at our example, if a house (1) is outside "Inner city" and (2) its price is greater than $700k, we will guess that it is a Beach House.

## Building Decision Tree

Decision trees of both regression and classification type follow the same procedure.

{% figure caption: "Fig. 3. Questions to ask when building a tree" %}
![]({{'/assets/images/decision-tree-question.png'}})
{% endfigure %}

Fig. 3. illustrates the standard architecture of decision trees along with the questions that you may ask when constructing each section of the tree. In general, there are three main steps in building a tree that concern with the root node, internal nodes and leaf nodes.

__Step 1:__ Define a root node

__Step 2:__ Expand the tree (recursively)

__Step 3:__ Decide when to stop

Recall that our goal is to have final groups that are pure (i.e. all members of the group belong to the same class). Therefore, we need to define a __measure of purity__ to compare splitting options, which will guide us through all 3 steps listed above.

### Regression tree: Measure of purity

To decide which independent variable is the best to sort the data, we use Residual Sum of Square (RSS) as the measure of purity.

$$
RSS = \sum_{j}^{J}\sum_{i \in j}(y_i - \bar{y_j})^2
$$


{% figure caption: "Fig. 4. Distance to city and No of bedrooms to predict House price" %}
![]({{'/assets/images/decision-tree-boundary.png'}})
{% endfigure %}

### Classification tree: Measure of purity



* Ask questions

* Classify the data based on the answer

Example

Yes/No

Rank (score 1, 2)

- Leaves can be numeric or categorical

- Can have different questions on each side


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

