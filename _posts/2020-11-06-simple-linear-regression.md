---
layout: post
title:  "Simple Linear Regression: Least Squares VS. Gradient Descent"
date:   2020-11-06 00:04
categories: supervised-learning
tags: regression least-squares gradient-descent
---
> This post sheds light on the motivation of Simple Linear Regression and discusses the two optimization methods Least Squares VS. Gradient Descent. Formulas derivation is also included at the end of the post as a bonus.
 
<!--more-->
It is human nature to be certain about our future and to make predictions based on existing clues. Hence, the idea of studying about the __association__ among subjects interests us. The first entry of the blog is dedicated to such an idea, it is about one basic concept in supervised learning: __Simple Linear Regression__.
 
{: class="table-of-content"}
* TOC
{:toc}
 
## Motivation
 
Simple Linear Regression assumes that the independent $$(x)$$ and dependent $$(y)$$ variables have a linear relationship. Our estimation of relationship is summarized in the formula below:
 
$$\hat{y} = \beta_0 + \beta_1x$$
 
$$\beta_0$$ is the intercept and $$\beta_1$$ is the coefficient (or slope). The name _linear regression_ suggests that the relationship resembles a straight line.

{% figure caption:"Fig. 1. Visualization of a Simple Linear Regression problem" class:"width_400" %}
![SLR Regplot]({{ '/assets/images/slr-regplot.png' | relative_url }})
{% endfigure %}

 
However, there can hardly be a perfect line that fits all the observations. Hence, errors are expected and we should find a way to minimize them. These errors are called **Residual Sum of Squares (RSS)** which measures the total of *squared* differences between our predictions and the true values (the differences are squared to prevent negative and positive values from cancelling out each other). The formula for RSS is as follows:
 
$$
RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^{2} = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^{2}
$$
 
In this post, I will discuss two different approaches to find the "perfect" line by optimizing RSS. While Least Squares method provides a _one-shot_ solution for the problem, Gradient Descent _gradually adapts_ to it.
 
{% figure caption:"Fig. 2. Basic ideas of Least Squares method and Gradient Descent" %}
![Compare Least Squares and Gradient Descent]({{ '/assets/images/least-squares-vs-gradient-descent.png' | relative_url }})
{% endfigure %}

 
## Least Squares Method
 
The objective is to find values of $$\beta_0$$ and $$\beta_1$$ that minimize RSS. We know from calculus that by taking the first derivative of a function (e.g $$f(x)$$), and setting it to 0, we can solve for critical values that minimize $$f(x)$$. This method works because the RSS function is convex, therefore, it has one global minimum.
 
{% figure caption:"Fig. 3. 3D plot of Residual Sum of Squares (RSS) (Source: [Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/))" class:"width_400" %}
![3D plot of RSS]({{ '/assets/images/3d-rss-slr.png' | relative_url }})
{% endfigure %}
 
The details of how to derive the formulas are included in the [BONUS Section](## BONUS: Least Squares - Formula Derivation). To summarize, we can obtain $$\beta_0$$ and $$\beta_1$$ using:
 
$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$
 
$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^{2}}$$
 
By plugging in these two parameters, we can now find the perfect line to fit our data and predict future values: $$\hat{y}_i = \beta_0 + \beta_1 x_i$$.
 
 
**Advantages**
 
Using Least Squares method, we can calculate the exact values of $$\beta_0$$ and $$\beta_1$$ using all $$x$$ and $$y$$ values in **one shot**. The idea is **straightforward** and well-known. For Simple Linear Regression, applying this method is effective because we only deal with two variables. This method is suitable for smaller datasets.
 
**Disadvantages**
 
Least Squares method is **limited to only a few loss functions** of certain shapes (e.g. RSS is a convex function). This method is expensive to implement on multivariate problems and mega datasets when **complex matrix operations** involve (will be discussed later in Multiple Regression module). Therefore, instead of arriving at the bottom of the bowl (i.e. reaching the global minimum) in one go, we can go downhill gradually. That is the inspiration of Gradient Descent. 
 
 
## Gradient Descent Method
Gradient Descent is a more general concept compared to Least Squares as it can be used on various loss functions. In the application of Simple Linear Regression, we can also use Gradient Descent to optimize RSS. The idea of Gradient Descent, as its name suggests, is to update the parameters gradually until the result reaches a certain threshold or completes a desired number of iterations (i.e. _epoches_). You can set up a _for-loop_ in Python to run Gradient Descent using the formula:
 
$$
\beta_k = \beta_k - \alpha \times \frac{\partial RSS}{\partial \beta_k}  \;\; ; \;\; k =0, 1
$$
 
**Advantages**
 
Gradient Descent works on a **diversity of loss functions**. Because of its versatility, Gradient Descent is a popular optimization technique used in many machine learning applications. Another benefit is that the computation step of updating the parameter is quite simple (usually the difficult part of complex models is to calculate the derivative of the loss function with respect to the parameters). Therefore, when dealing with multivariate problems such as Multiple Regression for **large datasets**, Gradient Descent has an edge over Least Square method (by not having to calculate the matrix inverses - we will discuss this in later entries).
 
**Disadvantages**

{% figure caption:"Fig. 4. The choice of Learning rate can greatly affect Gradient Descent performance." %}
![Learning rates in Gradient Descent]({{ '/assets/images/learning-rate-gradient-descent.png' | relative_url }})
{% endfigure %}
 
Choosing the hyperparameter - **learning rate ($$\alpha$$) can be a trouble** in Gradient Descent. Learning rate determines the step size of each iteration. If $$\alpha$$ is too large, there will be lots of deviation and it is difficult to hit the global minimum, or cannot hit the spot at all! If $$\alpha$$ is too small, it will take a long time to reach the global minimum. Learning rate is a hyperparameter meaning that we need to allocate a value for it. _"Is there a perfect learning rate?"_ - you may ask, it **depends on the problem**, but some popular choices are 0.1, 0.05, 0.01 or smaller values, but we don't always use a constant learning rate to run the whole model. For instance, in deep learning applications, we usually use learning rate decay (e.g. divide the learning rate in half after every 50 iterations) to prevent the gradients from exploding or vanishing. But for now, let's look at *Fig. 4* to see what different choices of learning rate look like in Simple Linear Regression.
 
 
## Model Evaluation
 
After fitting the model, we should check whether we have done a good job or if there are any mistakes. To assess the accuracy of the model, we will discuss two related quantities: Residual Standard Error (RSE) and Coefficient of Determination ($$R^{2}$$) {% cite james2013introduction %}.
 
### Residual Standard Error (RSE)
RSE is the measurement for the _lack of fit_. Although we have applied discussed techniques to achieve the "best" line, we would not be able to fit that line through all data points. This is because the relationship between $$X$$ and $$Y$$ are not perfectly linear. Comparing the _true_ equation $$y = \beta_0 + \beta_1x + \epsilon$$ to our _estimated_ equation $$y = \beta_0 + \beta_1x$$, it is clear that the error term $$\epsilon$$ is not covered.
 
$$RSE = \sqrt{\frac{1}{n-2}RSS} = \sqrt{\frac{1}{n-2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^{2}}$$
 
If $$y_i \approx \hat{y}_i$$, which indicates that our prediction is close to the true value, RSE will be small and we can conclude that our model is a good fit for the data. On the other hand, if the model is not a good fit, $$(y_i - \hat{y}_i)^{2}$$ will be large and so as RSE.
 
 
### Coefficient of Determination ($$R^{2}$$)
On the other end of the spectrum, $$R^{2}$$ is the measurement of _model good fit_. It can be interpreted as __the amount of variations in $$Y$$ that can be explained by $$X$$__ using our model.
 
$$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^{2}}{\sum_{i=1}^{n}(y_i - \bar{y})^{2}}$$
 
RSS is the variations left in $$Y$$ after fitting our model, this is how much the model is lacking. TSS is the total variations (or the natural variations) in $$Y$$ before fitting the model. Therefore, $$R^2$$ shows how much variations in $$Y$$ are resolved by our model. $$R^2$$ score lies between 0 and 1. For example, $$R^2 = 0.75$$ means that $$75\%$$ of the variations in $$Y$$ is explained by our model. A higher value of $$R^2$$ indicates great goodness of fit and vice versa.
 
 
## Summary
### Key Takeaways
In this post, we cover the core ideas of Simple Linear Regression and discuss two methods to optimize this model. The Least Squares method is straightforward and allows us to find $$\beta_0$$ and $$\beta_1$$ directly using two formulas. Gradient Descent can be used for a diversity of loss functions and can arrive at the solution gradually as long as we set a reasonable learning rate.
 
After optimizing the model, we can assess its accuracy using RSE and $$R^2$$, which represents the lack of fit and goodness of fit respectively. Thus, a good model should have low $$RSE$$ and high $$R^2$$.
 
### Limitation
Though the simplicity of the model is a good point, Simple Linear Regression has many shortcomings. Today, we will briefly look at two major limitations. Firstly, Simple Linear Regression has an oversimplified assumption. In reality, many relationships are non-linear. Hence, forcing the relationship to be linear will not produce accurate predictions. Secondly, there can be many contributing factors to a problem and they can also have some interaction effects. Therefore, instead of having multiple Simple Linear Regression models, there should be an upgrade that takes into account the _synergy_ among variables. To answer that, we will consider __Multiple Linear Regression__ in the next entry. 
 
__Chloe's End Note__
> Thank you so much for checking out my first post. Please let me know if you have any ideas on improving this post. As a bonus, I have included the details on how to derive Least Squares formulas. Hope that they will be helpful to someone. Also, stay tuned for my future posts :D
 
 
## BONUS: Least Squares - Formula Derivation
**Objective:** Find values of $$\beta_0$$ and $$\beta_1$$ that minimize RSS:
 
$$
RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^{2} = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^{2}
$$
 
### Find $$\beta_0$$
First, find the partial derivative of RSS with respect to $$\beta_0$$ treating $$x, y$$ and $$\beta_1$$ as constants:
 
$$\frac{\partial RSS}{\partial \beta_0} = -2 \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i)$$
 
Next, we set this value to 0 and solve for $$\beta_0$$:
 
$$
\begin{aligned}
-2 \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i) & = 0 \\
\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i) & = 0 \\
\sum_{i=1}^{n}y_i - n\beta_0 - n\beta_1 \sum_{i=1}^{n}x_i & = 0 \\
\end{aligned}
$$
 
$$\beta_0 = \frac{\sum_{i=1}^{n}y_i - \beta_1 \sum_{i=1}^{n}x_i}{n} $$      
 
Because:  $$\frac{\sum_{i=1}^{n}x_i}{n} = \bar{x}$$   and    $$\frac{\sum_{i=1}^{n}y_i}{n} = \bar{y}$$:
 
$${\color{Blue} {\beta_0 = \bar{y} - \beta_1 \bar{x}}}$$
 
 
### Find $$\beta_1$$
Similarly, we find the partial derivative of RSS with respect to $$\beta_1$$ and set it to 0:

$$
\begin{aligned}
\frac{\partial RSS}{\partial \beta_1} = -2 \sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1x_i) & = 0 \\
\sum_{i=1}^{n}x_iy_i - \beta_0 \sum_{i=1}^{n}x_i - \beta_1 \sum_{i=1}^{n}x_i^{2} & = 0
\end{aligned}
$$
 
Plug in $$\beta_0 = \bar{y} - \beta_1 \bar{x}$$:

$$
\begin{aligned}
\sum_{i=1}^{n}x_iy_i - (\bar{y} - \beta_1 \bar{x})\sum_{i=1}^{n}x_i - \beta_1\sum_{i=1}^{n}x_i^{2} & = 0 \\
\sum_{i=1}^{n}x_iy_i - \bar{y}\sum_{i=1}^{n}x_i + \beta_1 \bar{x}\sum_{i=1}^{n}x_i - \beta_1\sum_{i=1}^{n}x_i^{2} & = 0 \\
\sum_{i=1}^{n}x_i(y_i - \bar{y}) - \beta_1 \sum_{i=1}^{n}x_i (x_i - \bar{x}) & = 0 \\
\end{aligned}
$$
 
Solve for $$\beta_1$$:
 
$$\beta_1 = \frac{\sum_{i=1}^{n}x_i(y_i - \bar{y})}{\sum_{i=1}^{n}x_i (x_i - \bar{x})}$$
 
We can conclude here and use this equation to find $$\beta_1$$, but the more widely used variation in books and lecture notes is:
 
$$\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^{2}} = \frac{S_{XY}}{S_{XX}}$$
 
Many people find the equation above easier to remember as its numerator $$(S_{XY})$$ and denominator $$(S_{XX})$$ are the outputs of some software functions. With a few algebra adjustments, we can transform our original equation into its well-known form:
 
$$
\begin{aligned}
\beta_1 
&= \frac{\sum_{i=1}^{n}x_i(y_i - \bar{y})}{\sum_{i=1}^{n}x_i (x_i - \bar{x})} \\
&= \frac{\sum_{i=1}^{n}x_i(y_i - \bar{y}) - \sum_{i=1}^{n} \bar{x} (y_i - \bar{y}) + \sum_{i=1}^{n} \bar{x} (y_i - \bar{y})}{\sum_{i=1}^{n}x_i (x_i - \bar{x}) - \sum_{i=1}^{n}\bar{x}(x_i - \bar{x}) + \sum_{i=1}^{n}\bar{x}(x_i - \bar{x})} \\
&= \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) + \bar{x} (\sum_{i=1}^{n}y_i - n\bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^{2} + \bar{x} (\sum_{i=1}^{n}x_i - n\bar{x})}
\end{aligned}
$$
 
__Note__: In the last term of the equation in the numerator, (1) $$\bar{x}$$ is a number, so it can come out of the sum and (2) $$\sum_{i=1}^{n}y_i = n\bar{y}$$. Therefore, $$ \bar{x} (\sum_{i=1}^{n}y_i - n\bar{y}) = 0$$. The same thing applies to the denominator, we have the final equation:
 
$${\color{Blue} {\beta_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^{2}}}}$$

## Reference

{% bibliography --cited %}