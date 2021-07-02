---
layout: post
title:  "Kaggle Challenge - Predict Housing Prices"
date:   2021-11-29 15:46
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

- Categorical columns:

\begin{enumerate}
  \item MSSubClass
  \item MSZoning
  \item Street
  \item Alley
  \item LotShape
  \item LandContour
  \item Utilities
  \item LotConfig
  \item LandSlope
  \item Neighborhood
  \item Condition1
  \item Condition2
  \item BldgType
  \item HouseStyle
  \item RoofStyle
  \item RoofMatl
  \item Exterior1st
  \item Exterior2nd
  \item MasVnrType
  \item ExterQual
  \item ExterCond
  \item Foundation
  \item BsmtQual
  \item BsmtCond
  \item BsmtExposure
  \item BsmtFinType1
  \item BsmtFinType2
  \item Heating
  \item HeatingQC
  \item CentralAir
  \item Electrical
  \item KitchenQual
  \item Functional
  \item FireplaceQu
  \item GarageType
  \item GarageFinish
  \item GarageQual
  \item GarageCond
  \item PavedDrive
  \item PoolQC
  \item Fence
  \item MiscFeature
  \item SaleType
  \item SaleCondition
\end{enumerate}


__Check distribution of numerical fields__


__Check distribution of categorical fields__