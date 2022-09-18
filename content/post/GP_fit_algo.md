+++
title = "Gaussian Process Classification"
date = "2022-09-18"
author = "Ragnar Levi Gudmundarson"
tags = ["Classification", "Kernel"]
+++



In this notebook I will be trying to fit a Gaussian Process classification from scratch on very simple simulated data. I will use the laplacian, expectation propagation, variational inference (which will need some rework) and MH MCMC.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
import tqdm

```

Create dataset


```python
x = np.array(list(range(100)))
y = np.ones(100)
y[(x> 25) & (x<60)] = 0


#plt.scatter(x,y)
train_index = np.random.choice(list(range(100)), 20, replace=False)
train_index = np.sort(train_index)
y_train = y[train_index]
y_test= y[~np.isin(list(range(100)), train_index)]
x_train = x[train_index]
x_test = x[~np.isin(list(range(100)), train_index)]
plt.scatter(x_train,y_train)
print(x_train.shape)
print(x_test.shape)

```

    (20,)
    (80,)
    


    
![png](GP_fit_algo_2_1.png)
    


The problem with Gaussian Process classification is that the posterior

$$p(f_{new} | X, y,x_{new}) = \int p(f_{new} | X,x_{new}, f) p(f | X,y ) df$$

,where $f_{new}$ is the test function and is $x_{new}$ is the test input ,is not tractable . We will look at methods that approximate the psoterior to make the integral tractable. We find some distribution $q$ to be close to $p(f | X,y )$. We know of to deal with Guassian so we assume $q$ is Gaussian.

Once we have found $q$ we can approximate the integral by replacing $p(f|x,y)$ with $q(f|x,y;\mu, \Sigma)$ (once we have found $\mu$ and $\Sigma$). We know that the two Gaussian integrated together will be another Gaussian with mean.

$$E[f_{new} | X, y,x_{new}] = E[E[f_{new} | X, y,x_{new},f]| X, y,x_{new}] = E[K_{new}K^{-1}f| X, y,x_{new}] = K_{new}K^{-1}\mu$$

and variance

$$
\begin{split}
V[f_{new} | X, y,x_{new}]  &= E[V[f_{new} | X, y,x_{new},f]| X, y,x_{new}] + V[E[f_{new} | X, y,x_{new},f]| X, y,x_{new}] \\\\
&= E[K_{newnew} -K_{new}K^{-1}K_{new}^T| X, y,x_{new}] +V[K_{new}K^{-1}f| X, y,x_{new}] \\\\
&= K_{newnew} -K_{new}K^{-1}K_{new}^T + K_{new}K^{-1}\Sigma K^{-1}K_{new}^T
\end{split}
$$

where $K_{newnew} = K(X_{new},X_{new})$ and $K_{new} = K(X,X_{new})$



Functions for the sigmoid, log-likelihood. Also calcualte the kernel assuming a RBF kernel. Just set some gamma parameter that should be good enough


```python
gamma = 0.005
#from scipy.special import expit as sigmoid

def sigma(f):

    return 1.0/(1.0+np.exp(-f))

def log_lik(y,f):
    return - np.log(1+np.exp(-y*f))

K = rbf_kernel(np.expand_dims(x_train,1), gamma = gamma)
sns.heatmap(K)

Kstar = rbf_kernel(np.expand_dims(x_train,1),np.expand_dims(x_test,1), gamma = gamma)
Ktest = rbf_kernel(np.expand_dims(x_test,1), gamma = gamma)
print(Ktest.shape)
sns.heatmap(K)
```





    
![png](GP_fit_algo_5_2.png)
    


# Laplace approximation


```python
def fit_laplace(y,K):

    f = np.zeros(len(y))

    for _ in range(100):
        t = y
        W = -np.diag(-sigma(f)*(1-sigma(f)))
        d = t - sigma(f)
        B = np.identity(W.shape[0]) + np.dot(np.sqrt(W), K).dot(np.sqrt(W))
        L = np.linalg.cholesky(B)
        B_inv = np.dot(np.linalg.inv(L.T),np.linalg.inv(L))
        b = np.dot(W,f) + d
        a = b - np.dot(np.sqrt(W), B_inv).dot(np.sqrt(W)).dot(K).dot(b)
        f = np.dot(K,a)

        #f = np.dot(np.linalg.inv(np.linalg.inv(K)+W), np.dot(W,f) + d)

        #if _ %10 == 0:
            #print(-0.5*np.inner(a,f) + np.sum(log_lik(y,f)) - np.sum(np.log(np.diag(L))))

    return f

f = fit_laplace(y_train, K)
```

Train fit


```python
plt.plot(x_train,sigma(f))
plt.scatter(x_train, y_train)
```





    
![png](GP_fit_algo_9_1.png)
    


MAP prediction using $\hat{\pi} = \sigma(E(f_{new}))$, where $E(f_{new}) = k_{new}^{T}K^{-1}\hat{f} = k_{new}^{T} \nabla \log p(y |f)$


```python
d= y_train - sigma(f)
y_pred = sigma(np.dot(Kstar.T, d))
plt.plot(x_test,y_pred)
plt.scatter(x_test, y_test)
```





    
![png](GP_fit_algo_11_1.png)
    


We can also predict using averaged predictive probability, $\hat{\pi} = \int \sigma(f_{new}) q(f_{new},X,y,x_{new}) df_{new}$, the integral is not tractable so the logistic is usally approximated by the probit.


```python
W = -np.diag(-sigma(f)*(1-sigma(f)))
B = np.identity(W.shape[0]) + np.dot(np.sqrt(W), K).dot(np.sqrt(W))
L = np.linalg.cholesky(B)
f_star = np.dot(Kstar.T, d)
# need variance as well
middle_part = np.sqrt(W).dot(np.linalg.inv(L.T)).dot(np.linalg.inv(L)).dot(np.sqrt(W))
var_f = np.diag(Ktest) - np.diag(np.dot(Kstar.T, middle_part).dot(Kstar))
# probit approximation
lamda_sq = np.pi/8.0
kappa = 1 / np.sqrt(1.0/lamda_sq +  var_f)
y_pred_laplace = sigma(kappa * f_star)
plt.plot(x_test,y_pred_laplace)
plt.scatter(x_test, y_test)


```



    
![png](GP_fit_algo_13_1.png)
    


We can also plot the latent $f$


```python
plt.plot(x_test, f_star, color='green')
plt.fill_between(x_test.ravel(), 
                 f_star + 2*np.sqrt(var_f), 
                 f_star - 2*np.sqrt(var_f), 
                 color='lightblue')
```






    
![png](GP_fit_algo_15_1.png)
    


# Expectation Propagation

We want to approximate by approximating the individual likelihoods with a normal

$$ p(y_i|f_i) \approx t_i(f_i) = N(f_i| \tilde{\mu}_i, \tilde{\sigma}_i) $$

Call $q(f|\mu, \Sigma)$ the approximation of the posterior $p(f|X,y)$. The EP algorithm does the following:

* Message elimination: Choose a $t_i$ to perform an approximation. Create a cavity distribution by removing $t_i$ from current approximation $q^{-i}= \frac{q}{t_i}$
* Belief projection: Find $q_{new}$ that minimizes $KL(\hat{p}|| q)$, where $\hat{p}(f_i) \propto  q^{-i}(f_i) p(y_i|f_i)$. As $q(f_i)$ is normal this is equal to moment matching.
* Message update: Calcuate the new approximatiing factor $t_i \propto \frac{q_{new}}{q^{-i}}$



Define $\tilde{\tau}_{i}=\tilde{\sigma} _{i}^{-2}$, $\tilde{\nu}= \tilde{S}\tilde{\mu}$, $\tilde{S} = diag(\tilde{\tau})$, $\tilde{\tau} _{-i} = \tilde{\sigma} _{-i}^{-2}$ and $\tilde{\nu} _{-i} = \tilde{\tau} _{-i} \tilde{\mu} _{-i}$


```python
def ep_gp(K, y):
    y_ = y.copy()
    y_[y_ == 0] = -1

    # initialize parameters of q
    n = K.shape[0]
    nu_tilde = np.zeros(n)
    tau_tilde = np.zeros(n)
    mu = np.zeros(n)
    Sigma = K.copy() #+ np.identity(n)

    for i_outer in range(20):
        for j in range(n):
            # cavity parameters
            tau_cav = np.reciprocal(Sigma[j,j]) - tau_tilde[j]
            nu_cav =  np.reciprocal(Sigma[j,j])*mu[j] - nu_tilde[j]
            mu_cav = nu_cav/tau_cav
            sigma_cav = 1/tau_cav
            # match moment
            z = (y_[j]*mu_cav)/np.sqrt(1+sigma_cav)
            mu_hat = mu_cav  + y_[j]*sigma_cav*norm.pdf(z)/(norm.cdf(z)*np.sqrt(1+sigma_cav))
            sigma_hat = sigma_cav - sigma_cav**2 * norm.pdf(z) *(z + norm.pdf(z)/norm.cdf(z))/((1+sigma_cav)*norm.cdf(z))
            # Message update
            delta_tau = np.reciprocal(sigma_hat) - tau_cav - tau_tilde[j]
            tau_tilde[j] = tau_tilde[j] + delta_tau 
            nu_tilde[j] = np.reciprocal(sigma_hat)*mu_hat - nu_cav
            Sigma = Sigma - np.reciprocal((np.reciprocal(delta_tau) + Sigma[j,j]))*np.outer(Sigma[:,j],Sigma[:,j])
            mu = np.dot(Sigma,nu_tilde)

        S = np.diag(tau_tilde)
        L = np.linalg.cholesky(np.identity(n) + np.dot(np.sqrt(S),K).dot(np.sqrt(S)))
        V = np.dot(np.linalg.inv(L.T), np.dot(np.sqrt(S), K))
        Sigma = K - np.dot(V.T, V)
        mu = np.dot(Sigma, nu_tilde)

    return nu_tilde, tau_tilde

nu_tilde, tau_tilde = ep_gp(K, y_train)



```


With these parameters we can get the mean of the latent gaussian process (train) using the same equations as for "normal" gaussian processes as we have approximated the likelihood with a normal


```python
# get mean of latent gp
mu_tilde = np.dot(np.diag(np.reciprocal(tau_tilde)), nu_tilde)
S = np.diag(tau_tilde)
B = np.identity(K.shape[0]) + np.dot(np.sqrt(S),K).dot(np.sqrt(S)).dot(np.linalg.inv(B)).dot(np.sqrt(S)).dot(K)
Sigma = K - np.dot(K,np.sqrt(S))
f_mean = np.dot(Sigma, np.diag(tau_tilde)).dot(mu_tilde)

plt.plot(x_train, f_mean)

```



    
![png](GP_fit_algo_21_1.png)
    



```python
plt.plot(x_train,sigma(f_mean))
plt.scatter(x_train, y_train)
```


    
![png](GP_fit_algo_22_1.png)
    


And we can predict using for example average probabilities 



```python
def predict_ep(nu, tau, K, Kstar, Ktest):
    n = K.shape[0]

    S = np.diag(tau_tilde)
    L = np.linalg.cholesky(np.identity(n) + np.dot(np.sqrt(S),K).dot(np.sqrt(S)))

    z = np.dot(np.sqrt(S), np.linalg.inv(L.T)).dot(np.linalg.inv(L)).dot(np.sqrt(S)).dot(K).dot(nu)
    fstar = np.dot(Kstar.T,nu - z)
    v = np.dot(np.linalg.inv(L), np.sqrt(S)).dot(Kstar)
    V = np.diag(Ktest) - np.diag(np.dot(v.T,v))

    return norm.cdf(fstar/np.sqrt(1+V))

y_pred_ep = predict_ep(nu_tilde, tau_tilde, K, Kstar, Ktest)
plt.plot(x_test,y_pred_ep)
plt.scatter(x_test, y_test)
```



    
![png](GP_fit_algo_25_1.png)
    


# Variational Inference

We wan to maximize the elbow

$$ELBO = \int \log p(y|f)q(f)df - \text{KL}[q(f)||p(f)]$$

the KL term is analytical as $q$ and $p$ are Gaussians. The likelihood is harder. It is possible to write
$$
\begin{split}
\int \log p(y|f)q(f)df = \sum_i \int p(y_i|f_i)q(f_i)df 
\end{split}
$$

which we can approximate with a Gaussain quadrature. This code is not good, might look into this at a future time again.





```python

def ELBO(y, Sigma, mu, K):

    # Gaussain quadrature.
    integral = 0.0
    for i in range(len(y)):
        u = np.linspace(-4,4,10000)
        t = u/np.sqrt(2)
        w = np.exp(-t**2)/np.sqrt(np.pi)
        x = np.sqrt(2)*Sigma[i,i]*t + mu[i]
        ratio = np.exp(x)/(1+np.exp(x))
        F_t = y[i]*np.log(ratio) + (1-y[i])*np.log(1-ratio) 
        integral += np.sum(w*F_t)/10000

    # KL
    n = K.shape[0]
    I = 1e-5*np.identity(n)
    K_inv= np.linalg.inv(K+I)
    n = float(n)
    KL = 0.5*(np.log(np.linalg.det(K + I)) - np.log(np.linalg.det(Sigma + I)) - n + np.trace(np.dot(K_inv, Sigma)) + np.dot(mu, K_inv).dot(mu) )

    print(integral - KL)
    return (integral - KL)


def obj_fun_elbo(par, y,K):
    n = K.shape[0]
    #Sigma = np.zeros((n,n))
    #Sigma[np.triu_indices(n)] = par[:int(n*(n+1)/2)]
    #Sigma = np.triu(Sigma) + np.triu(K, 1).T
    #mu = par[int(n*(n+1)/2):]

    Sigma = np.diag(par[:n]**2)
    mu = par[n:]

    l, _ = np.linalg.eigh(Sigma)
    assert np.all(l >0)

    return -ELBO(y, Sigma, mu, K)




```

Perform minimization, this will take a while....


```python
n = K.shape[0]
sigma_0 = np.identity(n)
x0 = np.hstack([np.ones(n), np.zeros(n)])

out_var_inf = minimize(obj_fun_elbo, x0 = x0, args = (y_train,K), method = 'L-BFGS-B')
```



```python
plt.plot(x_train,out_var_inf.x[n:])
```





    
![png](GP_fit_algo_31_1.png)
    


Find predictive


```python
I = 1e-5*np.identity(n)
K_inv= np.linalg.inv(K+I)
f_var = np.dot(Kstar.T, K_inv).dot(out_var_inf.x[n:])
y_pred_vp = sigma(f_var)


plt.plot(x_test,y_pred_vp)
plt.scatter(x_test, y_test)
```


    
![png](GP_fit_algo_33_1.png)
    


mean clsoe to 0.5, might be some error here

# MCMC

Using MCMC we can sample local regions instead of iteratively drawing samples from each posterior conditional density $p(f_i| f_{i},y)$ seperately. If $f_k$ are function points in region $k$. Then we can propose values from the conditional GP prior $Q(f^{t}|f^{t-1})= p(f_k^{t} | f_{-k}^{t-1})$. The proposed points are accepted with probability $\min(1,A)$ where

$$
\begin{split}
A &= \frac{p(f^t | y)/Q(f^t|f^{t-1})}{p(f^{t-1} | y)/Q(f^{t-1}|f^{t})} \\\\
&= \frac{p(y|f_k^t,f_{-k}^{t-1})p(f_k^t,f_{-k}^{t-1})p(f_k^{t-1}|f_{-k}^{t-1})}{p(y|f_k^{t-1},f_{-k}^{t-1})p(f_k^{t-1},f_{-k}^{t-1})p(f_k^{t}|f_{-k}^{t-1})} \\\\
&= \frac{p(y|f_k^t,f_{-k}^{t-1})p(f_k^t|f_{-k}^{t-1})p(f_{-k}^t)p(f_k^{t-1}|f_{-k}^{t-1})}{p(y|f_k^{t-1},f_{-k}^{t-1})p(f_k^{t-1} | f_{-k}^{t-1})p(f_{-k}^{t-1})p(f_k^{t}|f_{-k}^{t-1})} \\\\
&= \frac{p(y|f_k^t,f_{-k}^{t-1})}{p(y|f_k^{t-1},f_{-k}^{t-1})} 
\end{split}
$$

I like to think that we are contitioning on $f_{-k}^t = f_{-k}^{t-1}$ and the we are taking $t \gets tn_k$ steps where $n_k$ is the number of regions.


```python
from scipy.stats import multivariate_normal
def bernoulli(y,x):
    return sigma(x)**y*(1-sigma(x))**(1-y)

n_t = len(y_train)
B = 10000

f = np.zeros((B+1,n_t))
r = np.zeros((B,n_t))  # how many accepted

index = np.arange(n_t)
for i in tqdm.tqdm(range(B)):
    for j in np.array_split(range(n_t),5):
        index_not_j = index[~np.isin(index,j)]
        K_inv = np.linalg.inv(K[np.ix_(index_not_j,index_not_j)] + 1e-2*np.identity(len(index_not_j)))  # bottlneck + regularization for condition
        # conditional mean 
        m = np.dot(K[np.ix_(j,index_not_j)],K_inv).dot(f[i, index_not_j])
        # conditional variance
        s = K[np.ix_(j,j)] - np.dot(K[np.ix_(j,index_not_j)],K_inv).dot(K[np.ix_(j,index_not_j)].T)
        f_new = f[i,:].copy()
        f_new[j] = multivariate_normal.rvs(mean = m, cov = s)
        y_lik_new = np.sum([np.log(bernoulli(y_train[k],f_new[k])) for k in range(n_t)])
        y_lik_old = np.sum([np.log(bernoulli(y_train[k],f[i,k])) for k in range(n_t)])
        
        if np.log(np.random.uniform()) <= np.min((y_lik_new - y_lik_old,0)):
            f[i+1,j] = f_new[j]
            r[i,j] = 1
        else:
            f[i+1,j] = f[i,j]


```

    100%|██████████| 10000/10000 [00:22<00:00, 449.70it/s]
    


```python
np.mean(r,0)
```




    array([0.7516, 0.7516, 0.7516, 0.7516, 0.8562, 0.8562, 0.8562, 0.8562,
           0.9313, 0.9313, 0.9313, 0.9313, 0.8495, 0.8495, 0.8495, 0.8495,
           0.505 , 0.505 , 0.505 , 0.505 ])



More rejections at end, might to increase acceptance we increase rejections at endpoints. We can also plot traceplot of one $f_i$


```python
plt.plot(range(B+1), f[:,3])
```


    
![png](GP_fit_algo_40_1.png)
    


Very autocorrelated, might have to do thinning, test other regions, increase regularization


```python
burning = 2000
f_latent_m = np.mean(f[burning:,],0)
f_latent_s = np.std(f[burning:,],0)

ci = 1.96*f_latent_s
fig, ax = plt.subplots(1,1)
ax.plot(x_train, f_latent_m)
ax.fill_between(x_train, (f_latent_m-ci), (f_latent_m + ci), color = 'b', alpha = .1)
```



    
![png](GP_fit_algo_42_1.png)
    


Instead of doing simulations $f(f_{new}|X,y,x_{new}) = \int p(f_{new}|X,x_{new},f)p(f|X,y)df $ we can approximatione the $f_{new}$ as $N(E[f], V[f])$.


```python
f_m_mcmc =   np.dot(Kstar.T,np.linalg.inv(K + 1e-3{new}np.identity(K.shape[0])).dot(f_latent_m))
plt.plot(x_test, sigma(f_m_mcmc))
plt.scatter(x_test, y_test)
```


    
![png](GP_fit_algo_44_1.png)
    


# Compare


```python
plt.plot(x_test,y_pred_ep, label = "EP")
plt.plot(x_test,y_pred_laplace, label = "Laplace")
plt.plot(x_test,y_pred_vp, label = "VP")
plt.plot(x_test,sigma(f_m_mcmc), label = "MCMC")
plt.scatter(x_test, y_test)
plt.legend()
```





    
![png](GP_fit_algo_46_1.png)
    

EP and VI give a little strange results, might be something wrong, especally with the VI which I know is very slow and assuming factoruized $q$ is not a very realistic assumption.


