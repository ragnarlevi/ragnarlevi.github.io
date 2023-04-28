+++
title = "Dependent Gaussian Processes"
date = "2022-06-30"
author = "Ragnar Levi Gudmundarson"
tags = ["Kernel"]
+++

# Dependent Gaussian Processes

In this workbook, I am going to reproduce the work of Phillip Boyle and Marcus Frean [Dependent Gaussian Processes](https://proceedings.neurips.cc/paper/2004/file/59eb5dd36914c29b299c84b7ddaf08ec-Paper.pdf).
I will assume that the reader is familiar with the basics of Gaussian Processes



```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Consider a device that operates on a continuous, real-valued input signal overtime $x(t)$ and emits a continuous real-valued output $y(t)$. This device is a linear time-invariant (LTI) filter if it is 
* Linear: If $x(t) = x_1(t)+x_2(t) then the output is $y(t) = y_1(t) + y_2(t)
* Time invariant: The output response to a shifted input $x(t+ \tau)$ is $y(t+ \tau)$

An LTI filter is completely characterized by its impulse response, $h(t)$, which is equivalent to the output when the filter is stimulated by a unit impulse $\delta(t)$. Given the impulse response, we can find the output of the filter in response to any finite input via convolution:

$$ y(t) = h(t) * x(t) = \int h(t-\tau)x(\tau)d\tau$$


When a linear filter is excited with Gaussian white noise $w(t)$, the covariance function of a zero-mean output process can be shown to be:

$$ cov(y(t), y(t') = \int h(\tau) h(t' -t +\tau)d\tau$$

Furthermore, this brings us to the main theme of this workbook, if $y_1(t)$ and $y_2(t) are two real-values outputs driven by the **same** white noise w(t) then their cross-covariance can be written as

$$ cov(y_1(t), y_2(t') = \int h_1(\tau) h_2(t' -t +\tau)d\tau$$

where $h_i$ is the filter of process $y_i$.

If we define a Gaussian filter 

$$ h(x) = b \exp \big( -0.5 ax^2 \big)$$

then the covariance and cross-covariance functions can be calculated explicitly:

$$ cov_{ii}(t-t') = \frac{\pi^{1/2} b_i^2}{\sqrt(a_i) }  \exp \big( -0.25 a(t-t')^2 \big) $$
$$ cov_{ij}(t-t') = \frac{(2\pi)^{1/2} b_i b_j}{\sqrt(a_i + a_j) }  \exp \big( -0.25 \frac{a_1 a_2}{a_1 + a_2}(t-t')^2 \big) $$

Note that we can define this for the case when $t$ is a vector as well.


To motivate why this formulation is useful. We show that we can use one process say $y_2$ at time $t$ to predict unobserved values of $y_1$ at time $t$

We start by simulating two dependent processes



```python
time = np.arange(60)

y_signal = np.sin(time*0.4)
y1 = y_signal[:30] + np.random.normal(scale = 0.1,size = 30)
y2 = y_signal[:30] + np.random.normal(scale = 0.1, size = 30)
x1 = time[:30]
x2 = time[:30]
plt.plot(x1, y1)
plt.plot(x2, y2)
```




    [<matplotlib.lines.Line2D at 0x1171c9a3460>]




    
![png](DepGP_5_1.png)
    


Define the kernels


```python
from sklearn.metrics import pairwise_distances

def C_ii(a,b, sigma, x,y):
    d = pairwise_distances(np.expand_dims(x, axis = 1),np.expand_dims(y, axis = 1)) ** 2
    return ((b**2)*np.power(np.pi, 0.5)*np.exp(-0.25*a*d) / np.sqrt(a)) + np.identity(d.shape[0])*sigma


def C_ii_pred(a,b, x,y):
    d = pairwise_distances(np.expand_dims(x, axis = 1),np.expand_dims(y, axis = 1)) ** 2
    return (b**2)*np.power(np.pi, 0.5)*np.exp(-0.25*a*d ) / np.sqrt(a) 

def C_ij(a1,a2,b1,b2, x,y):

    d = pairwise_distances(np.expand_dims(x, axis = 1),np.expand_dims(y, axis = 1)) ** 2

    const1 = (2*np.pi) ** 0.5
    const2 = np.sqrt(a1+a2)
    const3 = a1*a2/(a1+a2)

    return b1*b2*const1 * np.exp(-0.5*const3 * d)/const2

def C_ji(a1,a2,b1,b2, x,y):

    d = pairwise_distances(np.expand_dims(y, axis = 1),np.expand_dims(x, axis = 1)) **2

    const1 = (2*np.pi) ** 0.5
    const2 = np.sqrt(a1+a2)
    const3 = a1*a2/(a1+a2)

    return b1*b2*const1 * np.exp(-0.5*const3 * d)/const2


def neg_lik(C, y):
    from numpy.linalg import cholesky, det
    from scipy.linalg import solve_triangular


    try:
        L = cholesky(C)
    except np.linalg.LinAlgError:
        return  9999
        
    S1 = solve_triangular(L, y, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)
    
    return np.sum(np.log(np.diagonal(L))) + \
            0.5 * y.dot(S2) + \
            0.5 * len(y) * np.log(2*np.pi)

    #return 0.5 * np.log(np.linalg.det(C)) + \
    #           0.5 * np.dot(y,np.linalg.inv(C)).dot(y) + \
    #           0.5 * len(y) * np.log(2*np.pi)

    # return const1 + const2 + const3

def predict(param, y1, y2, x1,x2, x1_star, x2_star, sigma):
    a1 = param[0]
    a2 = param[1]
    b1 = param[2]
    b2 = param[3]
    C_11 = C_ii(a1, b1, sigma,x1,x1)
    C_12 = C_ij(a1, a2,b1,b2,x1,x2)
    C_21 = C_ji(a1, a2,b1,b2,x1,x2)
    C_22 = C_ii(a2, b2, sigma,x2,x2)


    C = np.block([[C_11, C_12], [C_21, C_22]])
    y = np.concatenate((y1,y2))

    C_11_star = C_ii_pred(a1, b1,x1_star,x1)
    C_12_star = C_ij(a1, a2,b1,b2,x1_star,x2)
    C_21_star = C_ji(a1, a2, b1,b2,x1,x2_star)
    C_22_star = C_ii_pred(a2, b2,x2_star,x2)



    C_star = np.block([[C_11_star, C_12_star], [C_21_star, C_22_star]])



    return np.matmul(C_star,np.linalg.inv(C)).dot(y), C, C_star
    
a1 = 1
a2 = 1
b1 = 0.2
b2 = 0.2 #0.2
sigma = 0.1
C_11 = C_ii(a1, b1, sigma,x1,x1)
C_12 = C_ij(a1, a2, b1,b2,x1,x2)
C_21 = C_ji(a1, a2, b1,b2,x1,x2)
C_22 = C_ii(a2, b2, sigma,x2,x2)

C = np.block([[C_11, C_12], [C_21, C_22]])
y = np.concatenate((y1,y2))




y_pred,_,_ = predict((a1, a2, b1, b2), y1, y2, x1,x2, x1, x2, sigma)
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].scatter(x1, y1, label ='original' )
ax[0].plot(x1, y_pred[:30], label ='predicted' )
ax[0].legend()

ax[1].scatter(x2, y2, label ='original' )
ax[1].plot(x2, y_pred[30:], label ='predicted' )
ax[1].legend()

neg_lik(C,y)



```




    29.1429757365891




    
![png](DepGP_7_1.png)
    


Our kernels seem to be working. Next, we define a function to estimate the hyperparameters using the marginal log-likelihood. For the marginal case:


```python
from scipy.optimize import minimize

def objective_marginal(param, y1, x1, sigma):

    a1 = param[0]
    b1 = param[1]

    C_11 = C_ii(a1, b1,sigma,x1,x1)

    cost = neg_lik(C_11,y1)
    return cost

def predict_marginal(param, y1, x1, x1_star,  sigma):
    a1 = param[0]
    b1 = param[1]

    C_11 = C_ii(a1, b1, sigma,x1,x1)


    C_11_star = C_ii_pred(a1, b1,x1,x1_star)


    return np.matmul(C_11_star.T,np.linalg.inv(C_11)).dot(y1)


out = minimize(objective_marginal, x0 = [10,0.3], args=(y1,x1, 0.1),bounds=((1e-5, 100), (1e-5, 100)), method = 'L-BFGS-B')

y_pred = predict_marginal(out.x, y1, x1,x1, sigma)
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax.scatter(x1, y1, label ='original' )
ax.plot(x1, y_pred[:30], label ='predicted' )
ax.legend()
out.x

```




    array([0.12391961, 0.36522661])




    
![png](DepGP_9_1.png)
    


Seems to be working. Now we define a function to find the best hyperparameters of the joint process


```python
from scipy.optimize import minimize

def objective(param, y1, y2, x1,x2, sigma):

    a1 = param[0]
    a2 = param[1]
    b1 = param[2]
    b2 = param[3]

    C_11 = C_ii(a1, b1,sigma,x1,x1)
    C_12 = C_ij(a1, a2,b1,b2,x1,x2)
    C_21 = C_ji(a1, a2,b1,b2,x1,x2)
    C_22 = C_ii(a2, b2,sigma,x2,x2)

    C = np.block([[C_11, C_12], [C_21, C_22]])
    y = np.concatenate((y1,y2))


    n1 = len(y1)
    n2 = len(y2)

    cost = neg_lik(C,y)
    return cost


out = minimize(objective, x0 = [0.5,0.5, 0.1, 0.1], args=(y1,y2,x1,x2, 0.1),bounds=((1e-5, 100), (1e-5, 100), (1e-5, 100), (1e-5, 100)), method = 'L-BFGS-B')


y_pred,_,_ = predict(out.x, y1, y2, x1,x2,x1,x2, sigma)
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].scatter(x1, y1, label ='original' )
ax[0].plot(x1, y_pred[:30], label ='predicted' )
ax[0].legend()

ax[1].scatter(x2, y2, label ='original' )
ax[1].plot(x2, y_pred[30:], label ='predicted' )
ax[1].legend()

out.x

```




    array([0.11501486, 0.10188409, 0.3697313 , 0.40364407])




    
![png](DepGP_11_1.png)
    
Works. In the remaining part, we illustrate the usefulness of the framework. We mask a part of the data and try to predict it. First by using no information and then by using the information of process 2.


```python
y1_masked = np.concatenate((y1[:7], y1[20:]))
x1_masked =  np.concatenate((x1[:7], x1[20:]))
plt.scatter(x1_masked, y1_masked)
```




    <matplotlib.collections.PathCollection at 0x1171c85a0e0>




    
![png](DepGP_13_1.png)
    


Marginal case (no information):


```python

out_marginal = minimize(objective_marginal, x0 = [1,0.01], args=(y1_masked,x1_masked, 0.1),
bounds=((0.05, None), (0.05, None)),
 method = 'L-BFGS-B')
out_marginal

y_pred = predict_marginal(out_marginal.x, y1_masked, x1_masked,x1, 0.1)
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax.scatter(x1_masked, y1_masked, label ='Observed Data', marker= 'x' )
ax.scatter(x1, y1, label ='All Data',alpha = 0.3 )
ax.plot(x1, y_pred[:30], label ='predicted' )
ax.legend()
out.x

```




    array([0.11501486, 0.10188409, 0.3697313 , 0.40364407])




    
![png](DepGP_15_1.png)
    


We see that the predictions are bad at the interval where we do not observe data. Next, we use the joint framework to extract information from the second process to use in the prediction.


```python
sigma = 0.1
out = minimize(objective, x0 = [0.2,0.2, 0.01, 0.01], args=(y1_masked,y2,x1_masked,x2, sigma),bounds=((1e-5, 100), (1e-5, 100), (1e-5, 100), (1e-5, 100)), method = 'L-BFGS-B')

y_pred, C,_ = predict(out.x, y1_masked, y2, x1_masked,x2,x1,x2, sigma) #[0.1, 0.1, 0.1, 0.1]
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].scatter(x1_masked, y1_masked, label ='Observed Data', marker= 'x' )
ax[0].scatter(x1, y1, label ='All Data',alpha = 0.3 )
ax[0].plot(x1, y_pred[:30], label ='predicted' )
ax[0].legend(bbox_to_anchor = (0.35,0.2))

ax[1].scatter(x2, y2, label ='original' )
ax[1].plot(x2, y_pred[30:], label ='predicted' )
ax[1].legend(bbox_to_anchor = (0.35,0.2))

neg_lik(C,np.concatenate((y1_masked, y2)))
out.x
```

array([0.13167358, 0.11177868, 0.35959916, 0.39335209])
    

![png](DepGP_18_2.png)
    
We see that the prediction is a lot better


```python
sns.heatmap(C)
```




    
![png](DepGP_20_1.png)
    

