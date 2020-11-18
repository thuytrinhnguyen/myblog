---
layout: post
title: "Exploratory Data Analysis in Python"
date: 2020-11-11 14:08
category: data-analysis
tags: data-analysis visualization 
---

> This post will show you how to perform standard Exploratory Data Analysis using Pandas operations and visualization in Matplotlib and Seaborn libraries.

<!--more-->

Once received a dataset, is it a good idea to jump straight in and conduct some complicated statistical tests? What if you are given a massive dataset of millions records, how do you approach it? Just like meeting a stranger at your friend dinner party, we need to have a few conversations to understand each other's background, hobbies, etc. before becoming acquaintances or even close friends. This post will show you how to approach a dataset and get to know it well. Hopefully, after reading this post you can become friends with the data you have on hand.

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}

## Motivation

Exploratory data analysis (EDA) is the fundamental step before conducting any in-depth analysis. The purpose of (EDA) is to extract the characteristics and assess the potential of each field in the dataset.

Firstly, we can perform descriptive statistics to gain initial understanding of the data. Then we handle missing values to, hopefully, improve completeness and preserve information. Lastly, as the data may come in different formats, we should standardize values to aid further analysis. 


## Dataset

To demonstrate the steps of EDA, we will be using a sample dataset on fruit prices and other data such as the grades given by the seller, the stock quantities and the review scores given by customers. A sample preview of the data is shown below:

{% figure class:"width_300" %}
![Sample Dataset]({{'/assets/images/eda-data.png'}})
{% endfigure %}


## Checking Data Type

It is a good practice to check the data type of all columns before proceeding to further exploration. The method <code>df.dtypes</code> returns a list of column name and data types. For example, in your sample dataset, __Type__ and __Grade__ have <code>object</code> data type meaning these columns contain string values, while others have type <code>float64</code> or <code>int64</code>.

You can change the data type of any columns using <code>df['column_name'].astype()</code>.

## Descriptive Statistics

Next, we perform descriptive statistics to understand key attributes of each field (e.g. the minimum, maximum, most frequent values and the distribution of the data). Let's review some useful functions for numerical and categorical data. 

### Numerical Data

__General description__

The method <code>df.describe()</code> in __Pandas__ will present a summary of some standard metrics on numerical columns dropping any NaN values. The output table gives us a rough idea of the data critical values and its distribution. 

{% figure class:"width_250" %}
![df.describe output]({{'/assets/images/eda-code-des.png'}})
{% endfigure %}

__Correlation__

Examining the potential correlation among columns is a also good idea. We can achieve that using the method <code>df.corr()</code>, which measures the Pearson correlation values of numerical columns. The output in our example suggests a strong relationship between the prices and customer review scores ($$\rho \approx 0.836$$). 

{% figure class:"width_250" %}
![df.corr output]({{'/assets/images/eda-corr.png'}})
{% endfigure %}

__Visualizing relationships__

To gain a better understanding of the data, we can observe the trend of numerical variables using a _scatterplot_. A _regplot_ (regression plot) does the same thing but with a regression line to check if the relationship resemble a linear association. Note that these plots only picture an initial impression, we should conduct statistical tests to come to the final conclusion. We will be using __Seaborn__ library to plot the data. In our example, we see that the price and review of the fruits have a positive relationship. As the review scores increase, so do the prices. The _Regplot_ suggests that they have a linear relationship.

* __Scatter Plot__

```python 
sns.scatterplot(x='Review', y='Price', data=df)
plt.title('Review VS Price Scatterplot')
plt.show()
```
{% figure class:"width_300" %}
![Scatterplot]({{'/assets/images/eda-scatter.png'}})
{% endfigure %}


* __Regression Plot__

```python 
sns.regplot(x='Review', y='Price', data=df)
plt.title('Review VS Price Regression Plot')
plt.show()
```

{% figure class:"width_300" %}
![Regplot]({{'/assets/images/eda-regplot.png'}})
{% endfigure %}

### Categorical Data

__General description__

The method <code>describe()</code> can also be used for categorical data if you pass in an <code>include</code> argument. The result is a summary table for both categorical and numerical data. For example, in the <code>Type</code> column, there are 14 values of 11 unique values which indicates the existence of various wordings for the same object. 

{% figure class:"width_300" %}
![df.describe(include='all')]({{'/assets/images/eda-cat-describe.png'}})
{% endfigure %}

__Unique value counts__

To count the frequency of each unique value in a categorical column, we use <code>value_counts</code>. Column <code>Type</code> has many spelling errors which we can improve with some standardization methods in a leter section. On the other hand, column <code>Grade</code> has only 3 values (A, B, C) among which B is the most frequent one with 5 occurences. 
  

{% figure class:"width_400" %}
![Value Counts]({{'/assets/images/eda-value-counts.png'}})
{% endfigure %}

## Missing data

Missing data is expected in any real life datasets. To count the number of missing values in each column, call the method:  <code>df.isna().sum()</code>. There are three popular techniques to deal with missing values: (1) Replace by mean, (2) replace by mode and (3) do nothing.

### Replace by mean

To replace NaN values in numerical columns, we use the method <code>fillna()</code>. In this case, I want to compare the 2 columns of Price (with and without NaN values), therefore, I assign the filled values to the column __'Price Without NAs'__. If you wish to make changes to the original column, add the argument <code>inplace=True</code>.

<code>df['Price Without NAs'] = df['Price'].fillna(df['Price'].mean())</code>

{% figure class:"width_200" %}
![Price Full]({{'/assets/images/eda-price-full.png'}})
{% endfigure %}

### Replace by mode

Similarly, we can replace NaN values in categorical columns using <code>fillna()</code>. To access the mode, use <code>mode()[0]</code> which will return the most common value in that column.

<code>df['Grade Without NAs'] = df['Grade'].fillna(df['Grade'].mode()[0])</code>

{% figure class:"width_200" %}
![Grade Full]({{'/assets/images/eda-grade-full.png'}})
{% endfigure %}

### Do nothing

Ideally, replacing missing values with our "best guess" to help preserve the meaning of the data. In some cases, however, leaving missing values as they are, is a more reasonable option. Let me explain. Replacing by mean or by mode has a "neutralizing" effect on the data, hence, it can alter some characteristics of the dataset if the missing values occupy a large portion of the data.   

## Standardization

Standardization makes comparing data easier and helps us see some hidden characteristics of the data. There are various techniques to standardize both numerical and categorical data.

### Numerical Data

__Common methods to standardize values__

Simple standardization is to divide all values by the __maximum__, which results in data value 0-1. 

MinMax standardization applies $$\frac{x - min}{max - min}$$.

Z-score standardization is more popular in statistics, $$z = \frac{x - \mu}{\sigma}$$.

The results of 3 methods are different and should be chosen depending on the domains and applications of the project.

```python 
review_min = df['Review'].min()
review_max = df['Review'].max()
review_mean = df['Review'].mean()
review_std = df['Review'].std()

df['Review Simple Standardize'] = df['Review'] / review_max
df['Review Minmax Standardize'] = (df['Review'] - review_min) / review_max
df['Review Zscore Standardize'] = (df['Review'] - review_mean) / review_std
```
{% figure class:"width_500" %}
![Review Standardized]({{'/assets/images/eda-num-std.png'}})
{% endfigure %}

__Binning__

Binning is to group data points into intervals. In other words, we can convert numerical data to categorical groups (e.g. age groups, budget levels, ...). 

* __Histogram__

To decide the boundaries or intervals, we should first observe the trends using a histogram. We can use methods <code>displot()</code>, <code>displot()</code> or <code>catplot(..., kind='hist',...)</code> from __Seaborn__ to generate elegant looking histograms. Notice that you can play around with the number of <code>bins</code> to check how many intervals work best on your data. Fig. 1. illustrates 2 histograms with 10 bins (on the left) and 5 bins (on the right).

```python 
sns.distplot(df['Quantity'], bins=10)  # or bins=5
plt.title('Quantity Histogram')
plt.show()
```

{% figure caption: "Fig. 1. Histogram comparison between different bin sizes" %}
![Histogram]({{'/assets/images/eda-multi-hist.png'}})
{% endfigure %}

* __Pandas.cut__

Once you have selected the groups to sort our data, use the method <code>pd.cut</code> to section the data into categories. You can do it manually (i.e. by specifying the boundaries of each interval) or automatically (i.e. by dividing the sections equally). I will give an example of categorizing __Quantity__ values into 3 equal-interval bins which indicates levels of inventory.

```python 
quantity_bin = np.linspace(df['Quantity'].min(), df['Quantity'].max(), 4)
quantity_group = ['Low Stock', 'Medium Stock', 'High Stock']

df['Quantity Binned'] = pd.cut(df['Quantity'], bins=quantity_bin, labels=quantity_group, include_lowest=True)
```

{% figure class: "width_200" %}
![Binning]({{'/assets/images/eda-binning.png'}})
{% endfigure %}


### Categorical Data

__Standardizing wording variations__

From previous part, we notice that there are spelling mistakes and wording variations that we can improve. The method <code>df.replace()</code> is useful for this purpose. In our example, I standardize different types of fruit into: Apple, Orange and Lemon. I use <code>dict.fromkeys()</code> to replace multiple values at once. To replace a single value and save the changes to the original column, use <code>df.replace('value to replace', 'replacing value', inplace=True).

```python
df['Type Standardized'] = df['Type'].copy()
      
df['Type Standardized'].replace(dict.fromkeys(['apples', 'Apps', 'Appel'], 'Apple'), inplace=True)
df['Type Standardized'].replace(dict.fromkeys(['orange', 'navel', 'Oranges'], 'Orange'), inplace=True)
df['Type Standardized'].replace(dict.fromkeys(['pink lemons', 'lemon', 'lemmon'], 'Lemon'), inplace=True)
```
{% figure class: "width_200" %}
![Categorical Wording]({{'/assets/images/eda-type-standard.png'}})
{% endfigure %}

__Groupby__

What if we want to compare the price of Apple - Grade A and Apple - Grade C (i.e. difference among Grades) or Apple - Grade A and Orange - Grade A (i.e. difference among Fruits)? We can use the method <code>groupby</code> to categorize data. In our example, I build a table containing the average prices of fruits of each grades - think of it as a double filter! 

```python  
df_subset = df[['Grade', 'Type Standardized', 'Price Without NAs']]
df_grp = df_subset.groupby(['Grade', 'Type Standardized'], as_index=False).mean()
```

{% figure class: "width_300" %}
![Groupby]({{'/assets/images/eda-groupby.png'}})
{% endfigure %}

__Pivot Table__ 

A more effective way to display the grouped data we just created is to use __Pivot Table__. I run <code>pivot()</code> method on our grouped data <code>df_grp</code> and assign which fields to be the column and row of our pivot table. Looking much cleaner now!

```python 
df_pivot = df_grp.pivot(index='Grade', columns='Type Standardized')
```

{% figure class: "width_300" %}
![Pivot]({{'/assets/images/eda-pivot.png'}})
{% endfigure %}


## Summary

In this post, I have shown you some basic techniques in Exploratory Data Analysis - to gain understanding of the data. We explore several different aspects of data preprocessing (e.g. missing values, standardization, binning, grouping), data visualization (e.g. scatterplot, histogram) and their applications on numerical and categorical data. I hope that you have learned something new today. Please leave a comment if anything in the post needs correction.

Thank you very much and stay tuned for more meaningful content every week!

