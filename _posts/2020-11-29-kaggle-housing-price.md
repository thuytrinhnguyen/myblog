---
layout: post
title:  "Kaggle Challenge - Predict Housing Prices"
date:   2020-11-29 15:46
categories: 
tags: kaggle
---

> To document how I cope with m first Kaggle challenge

<!--more-->

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}

## Step 1: Explore descriptive statistics of all fields

__Check and fix dtypes__

- Cross check with `data_description.txt`

- Change the following columns from `int64` or `float64` to `category`:
    - `MSSubClass`
    - `YearBuilt`
    - `YearRemodAdd`
    - `GarageYrBlt`
    - `MoSold`
    - `YrSold`
    
- Change columns that are in `object` type into `category` type


__Check distribution of numerical fields__


__Check distribution of categorical fields__