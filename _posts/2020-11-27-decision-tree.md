---
layout: post
title:  "Decision Tree"
date:   2020-11-27 15:44
categories: supervised-learning
tags: decision-tree classification regression
---

> This post covers the fundamental concepts in Decision Tree for both regression and classification purposes. We will go through the splitting criterion for branching the trees and discuss the pros and cons of this model. So let's buckle up!

<!--more-->

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}


## Purpose

Decision tree is a popular technique in supervised learning. The full name of this technique is classification and regression tree (CART). From its name you can probably guess that Decision tree is capable of both __classification__ and __regression__ problems. 

__So, in general, what does Decision tree do?__

Decision tree builds a set of rules that partition the data repeatedly. The ultimate goal is to stratify observations into segments to either classify or predict values of future data points.

{% figure caption: "Fig. 1. Example of a regression tree" class: "width_600"%}
![]({{'/assets/images/decision-tree-reg.png'}})
{% endfigure %}

__Fig. 1__ demonstrates a regression tree to predict house prices where each leaf node (i.e. the last node on each branch) is the __average__ price of the houses matching the previous conditions. For example, the mean price of all houses that (1) locate in "Inner city" area and (2) have no more than 2 bedrooms, is $400K. Using this regression tree, if a future data points falls into this group, we can predict its price to be $400K.   

{% figure caption: "Fig. 2. Example of a classification tree" class: "width_600"%}
![]({{'/assets/images/decision-tree-classification.png'}})
{% endfigure %}

Classification trees are slightly different as their leaves are categories instead of numerical values. __Fig. 2__ shows that although the leaf nodes in classification trees cannot be numerical, we can still include numerical predictors in our tree as internal nodes (e.g. Price < $700K). Looking at our example, if a house (1) is outside "Inner city" and (2) its price is greater than $700k, we can guess that it is a Beach House.

## Building Decision Tree

Decision trees of both regression and classification type follow the same procedure.

{% figure caption: "Fig. 3. Questions to ask when building a tree" %}
![]({{'/assets/images/decision-tree-question.png'}})
{% endfigure %}

__Fig. 3__ illustrates the standard architecture of decision trees along with the questions that you may ask when constructing each section of the tree. In general, there are three main steps in building a tree that concern with the root node, internal nodes and leaf nodes.

- Step 1: Define a root node

- Step 2: Expand the tree (recursively)

- Step 3: Decide when to stop

Recall that our goal is to have final groups that are pure (i.e. all members of the group belong to the same class). Therefore, we need to define a __measure of purity__ to compare splitting options, which will guide us through all 3 steps listed above.

### Regression tree: Measure of purity

To decide which independent variable is the best to sort the data, we use Residual Sum of Square (RSS) as the measure of purity.

$$
RSS = \sum_{j}^{J}\sum_{i \in j}(y_i - \bar{y_j})^2
$$

where $$J$$ is the number of groups/regions and $$i$$ is the member data point in each region.

For example, we would like to predict housing prices based on two features: Distance to city (in kilometer - km) and number of bedrooms. __Fig. 4__ demonstrates how we can classify houses that are closer to the city (i.e. less than 30km away from city) to be of high price. Among those that are more than 30km away from the city, houses with more than 2 bedrooms have medium price. 

{% figure caption: "Fig. 4. Distance to city and No of bedrooms to predict House price" class: "width_500"%}
![]({{'/assets/images/decision-tree-br.png'}})
{% endfigure %}

Using the rules in __Fig. 4__, we have constructed 3 regions R1 - High, R2 - Medium and R3 - Low price as shown in __Fig. 5__.

{% figure caption: "Fig. 5. Housing price regions " class: "width_500"%}
![]({{'/assets/images/decision-tree-boundary.png'}})
{% endfigure %}

To measure the purity of the splits suggested in Fig 5, we first calculate the means of 3 regions. Then, in each region, we calculate $$(y_i - \bar{y_j})^2$$ for every member $$i$$ of group $$j$$. We repeat the process for all 3 regions. The total RSS of all 3 regions is the measure of purity for this splitting decision.

$$
RSS = \sum_{i \in R1}(y_i - \bar{y}_{R1})^2 + \sum_{i \in R2}(y_i - \bar{y}_{R2})^2 + \sum_{i \in R3}(y_i - \bar{y}_{R3})^2
$$

Our goal is to have high homogeneity among each group, hence, the optimal splitting criterion should result in a minimal RSS (i.e. having the least amount of impurity within the group). We will apply this rationale when deciding the variables to branch on. 

### Classification tree: Measure of purity

Two popular methods to measure impurity used in classification trees are: __Gini index__ and __[Entropy]({% post_url 2020-11-23-entropy %})__. __Table 1__ compares the differences between Gini and Entropy. Mathematically, they are very similar. When using Gini, we want the split to result in a lower Gini. A pure leaf has $$Gini = 0$$. When using Entropy, on the other hand, we want to reduce randomness by measuring the information gain after making the split. We will choose the predictor that provides the most information gain. 

_Table. 1. Gini VS Entropy_
<table class="tg" >
<thead>
  <tr>
    <th class="tg-0hd2"></th>
    <th class="tg-7kw4">Gini </th>
    <th class="tg-2kp4">Entropy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-jwk7">Formula</td>
    <td class="tg-0hd2">$$Gini = 1 - \sum_{i}^{C} (p_i)^2$$</td>
    <td class="tg-0hd2">$$H = - \sum_{i}^{C}p_i \times log_{2}(p_i)$$</td>
  </tr>
  <tr>
    <td class="tg-jwk7">Splitting<br>Criterion</td>
    <td class="tg-0hd2">- Calculated Gini for each leaf<br>- Choose splitting decision that has the minimum Gini:<br><span style="font-weight:400;font-style:normal">$$min\; (\; G_{before\_split}, G_{weighted, \; after\_split})$$</span></td>
    <td class="tg-0hd2">- Calculate Entropy for each leaf<br>- Choose splitting decision that maximize information gain:<br>$$max\; (\; H_{before\_split} - H_{weighted, \; after\_split})$$</td>
  </tr>
  <tr>
    <td class="tg-jwk7">Stop<br>Condition</td>
    <td class="tg-0hd2">1. Achieve all pure leaves<br>2. Stop when Gini before split is minimum.<br>No benefit in further splitting</td>
    <td class="tg-0hd2">Achieve all pure leaves<br>Entropy is zero</td>
  </tr>
</tbody>
</table>

\
We measure the purity of each split using the discussed methods to choose (1) the root node and (2) the next predictors to branch on. In general, we let the tree to grow fully (i.e. split until we achieve only pure nodes) and apply different __pruning techniques__ to prevent overfitting.

## Advantages and Disadvantages of Decision Tree

### Advantages

- __Easy to conduct.__ Decision tree is a simple concept to understand. Especially with the help of modern programming frameworks, we can build trees within a few lines of codes.
  
- __Explainability.__ Decision tree can be visualized effectively, hence, it is highly interpretable for non-specialist stakeholders. It is a useful tool to explain the decision-making process to management.

### Disadvantages

- __Overfitting.__ The main issue with decision trees is that as their objective is to achieve pure nodes, the trees can go very deep to adapt to training examples. Hence, it often has high variance i.e. low accuracy on testing dataset. There are various techniques that help ease overfitting problem of decision tree at the cost of some explainability. Future posts will discuss these in details.

- __Stability.__ Small changes in the dataset can alter the probability of classes. Since the structure of decision tree heavily based on the distribution of the classes, it is not a stable model. 

## Summary

In this post, we have gone through the basic ideas of Decision Tree: 

- Decision tree can work on both Regression and Classification problems.
- We build decision tree by comparing the measure of purity among predictors using Residual Sum of Squares, Gini index, Entropy, etc.
- Although decision tree is highly interpretable, it is unstable and can overfit the data easily.

To cope with the main issues of decision tree and enhance the power of this model, concepts such as bagging, pruning, Random Forrest, were introduced. Future posts will discuss these techniques to extend the idea of Decision Tree.
