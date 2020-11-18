---
layout: post
title:  "Data Visualization in Python"
date:   2020-11-30 00:04
categories: 
tags: 
---

__Boxplot__

```python 
sns.catplot(x='Type Standardized', y='Price Without NAs', kind='box', data=df, palette='Set2')
plt.title('Fruit Types VS Price Boxplot')
plt.xlabel('Fruit Types')
plt.ylabel('Price')
plt.show()
```

> Sneakpeek

<!--more-->

<mark><b>Highlights</b></mark> 

{: class="table-of-content"}
* TOC
{:toc}




<code>sns.boxplot(x='Type Standardized', y='Price Without NAs', data=df, palette='Set2')</code>

{% figure class: "width_300" %}
![Boxplot]({{'/assets/images/eda-boxplot.png'}})
{% endfigure %}

__Bar Chart__

```python 
sns.catplot(x='Price', y='Type Standardized', kind='bar', orient='h', ci=None, palette='Set2',  data=df)
plt.title('Fruit Types VS Price Bar Chart')
plt.xlabel('Fruit Types')
plt.ylabel('Price')
plt.show()
```

{% figure class: "width_300" %}
![Barchart]({{'/assets/images/eda-bar.png'}})
{% endfigure %}
