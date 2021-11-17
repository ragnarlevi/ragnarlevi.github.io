+++
title = "Quntitative Risk Analysis"
date = "2021-11-17"
author = "Ragnar Levi Gudmundarson"
tags = ["Risk", "Extremes"]
+++


International banking regulations require banks to pay specic attention to the
probability of large losses over short periods of time

Furthermore, use of the empirical distribution will mean that simulated future
observations can never exceed the maximum historical observation. This makes
simulation of extreme values problematic.

Underestimating the dependence among extreme risks can lead to serious consequences, as for instance those we experienced during the last financial crisis

Including extreme risks in probabilistic models is recognized nowadays as a necessary condition for good risk management in any institution, and not restricted anymore to reinsurance companies, who are the providers of covers for natural catastrophes


```python
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, kurtosis, skew, nct, beta, pareto, expon, genpareto, kendalltau
from scipy.optimize import minimize, brentq
from statsmodels.distributions.empirical_distribution import ECDF
```


```python
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
url_esg = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols=^GSPC&range=10y&interval=1d&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance"
response = requests.get(url_esg, headers=headers)
```


```python
def get_index(tick):
    """
    Function that takes the sp500 index from yahoo
    """
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    # ESG historical data (only changes yearly)
    url_esg = f"https://query1.finance.yahoo.com/v7/finance/spark?symbols={tick}&range=10y&interval=1d&indicators=close&includeTimestamps=false&includePrePost=false&corsDomain=finance.yahoo.com&.tsrc=finance"
    response = requests.get(url_esg, headers=headers)
    if response.ok:
        sp500 = pd.DataFrame({'date':pd.to_datetime(response.json()['spark']['result'][0]['response'][0]['timestamp'], unit= 's'),
                              'price':response.json()['spark']['result'][0]['response'][0]['indicators']['quote'][0]['close']})
    
    else:
        print("Empty data frame")
        sp500 = pd.DataFrame()



    return sp500
```


```python
sp500 = get_index('^GSPC')
nasdaq = get_index('^IXIC')
```


```python
plt.figure(figsize=(5,5))
plt.plot(sp500['date'], sp500['price'])
```




    [<matplotlib.lines.Line2D at 0x286fa8c8fa0>]




    
![png](extremes_5_1.png)
    


Let's plot the daily log-returns


```python
sp500['log_return'] = np.log(sp500['price']).diff()
sp500 = sp500.iloc[:,:].dropna(axis= 0)

nasdaq['log_return'] = np.log(nasdaq['price']).diff()
nasdaq = nasdaq.iloc[:,:].dropna(axis= 0)
```


```python
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].plot(sp500['date'], sp500['log_return'])
ax[1].plot(nasdaq['date'], nasdaq['log_return'])
```




    
![png](extremes_8_1.png)
    


# Extremes

Let's plot the histogram and try different probability distributions. 


```python
mu_normal = np.mean(sp500['log_return'])
var_normal = np.mean((sp500['log_return']-mu_normal) ** 2)
normal_fit = norm.pdf(sorted(sp500['log_return']), mu_normal, np.sqrt(var_normal))
```


```python
mu_normal
```




    0.0005225186337781287



## A normal distribution

For the normal distribution we simply use the well known maximum likelihood estimators for parameters.


```python

```


```python
bins = plt.hist(sp500['log_return'], bins = 100, density=True)
plt.plot(sorted(sp500['log_return']),normal_fit)
```




    
![png](extremes_15_1.png)
    



```python
fig, ax = plt.subplots()
ax.plot([-1,1], [-1,1], color = 'black')
ax.scatter(norm.ppf(np.array(range(sp500.shape[0]))/sp500.shape[0], mu_normal, np.sqrt(var_normal)),sorted(sp500['log_return']))
ax.set_xlim([-0.04, 0.04])
ax.set_ylim([-0.1, 0.1])

```



    
![png](extremes_16_1.png)
    


## A student t distribution

$$ r_t = m + sx_t$$

where $x_t$ is the standard t random variable. We can fit by a method of moments


```python
m = np.mean(sp500['log_return'])
v = (4*kurtosis(sp500['log_return']) - 6)/(kurtosis(sp500['log_return'])-3)
s = np.sqrt( (np.std(sp500['log_return']) ** 2)* (v-2) / v)

t_fit = t.pdf(sorted(sp500['log_return']), df = v, loc = m, scale = s)
```


```python
bins = plt.hist(sp500['log_return'], bins = 100, density=True, label = 'empirical')
plt.plot(sorted(sp500['log_return']),t_fit, label = 't-dist')
plt.plot(sorted(sp500['log_return']),normal_fit, label = 'norm-dist')
plt.legend()
```





    
![png](extremes_19_1.png)
    



```python
fig, ax = plt.subplots()
ax.plot([-1,1], [-1,1], color = 'black')
ax.scatter(t.ppf(np.array(range(sp500.shape[0]))/sp500.shape[0], df = v, loc = m, scale = s),sorted(sp500['log_return']))
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
```




    
![png](extremes_20_1.png)
    


## non-central-t-distribution

The skewness of the sp500 data is



```python
skew(sp500['log_return'])
```




    -0.9309407317689141



Thus, it is natural to consider a distribution that is not symmetric. We have to run a optimization algorithm


```python
params = nct.fit(sp500['log_return'], floc = 0)
nct_fit = nct.pdf(sorted(sp500['log_return']), df = params[0], nc = params[1], loc = params[2], scale = params[3])
```


```python
bins = plt.hist(sp500['log_return'], bins = 100, density=True, label = 'empirical')
plt.plot(sorted(sp500['log_return']),t_fit, label = 't-dist')
plt.plot(sorted(sp500['log_return']),normal_fit, label = 'normal')
plt.plot(sorted(sp500['log_return']),nct_fit, label = 'nct')
plt.legend()
```



    
![png](extremes_25_1.png)
    


So the fit seems to be good. Let's inspect the tails.


```python
empirical_cdf = ECDF(sorted(sp500['log_return']))
normal_cdf = norm.cdf(sorted(sp500['log_return']), mu_normal, np.sqrt(var_normal))
t_cdf = t.cdf(sorted(sp500['log_return']), df = v, loc = m, scale = s)
nct_cdf = nct.cdf(sorted(sp500['log_return']), df = params[0], nc = params[1], loc = params[2], scale = params[3])
```


```python
fig, ax = plt.subplots(figsize = (10,5))
ax.plot(sorted(sp500['log_return']), empirical_cdf(sorted(sp500['log_return'])), label = 'empirical')
ax.plot(sorted(sp500['log_return']), normal_cdf, label = 'normal')
plt.plot(sorted(sp500['log_return']), t_cdf, label = 't-dist')
plt.plot(sorted(sp500['log_return']), nct_cdf, label = 'nct')
plt.legend()
plt.xlim([-0.05, 0.06])
```


    
![png](extremes_28_1.png)
    



```python
fig, ax = plt.subplots(1,2, figsize = (15,5))

ax[0].plot(sorted(sp500['log_return']), empirical_cdf(sorted(sp500['log_return'])), label = 'empirical')
ax[0].plot(sorted(sp500['log_return']), normal_cdf, label = 'normal')
ax[0].plot(sorted(sp500['log_return']), t_cdf, label = 't-dist')
ax[0].plot(sorted(sp500['log_return']), nct_cdf, label = 'nct')
ax[0].legend()
ax[0].set_ylim([0.95, 1])
ax[0].set_xlim([0, 0.1])

ax[1].plot(sorted(sp500['log_return']), empirical_cdf(sorted(sp500['log_return'])), label = 'empirical')
ax[1].plot(sorted(sp500['log_return']), normal_cdf, label = 'normal')
ax[1].plot(sorted(sp500['log_return']), t_cdf, label = 't-dist')
ax[1].plot(sorted(sp500['log_return']), nct_cdf, label = 'nct')
ax[1].legend()
ax[1].set_ylim([0.99, 1])
ax[1].set_xlim([0, 0.1])
```



    
![png](extremes_29_1.png)
    


The nct seems to fit the center part the best, while the t-distribution seems to fit the tail the best.

## Order Statistics of the empirical distribution

Let's compare the case when we calculate the quantiles from the observed data vs the theoretical quantile. We will assume that the data comes from a t distribution with 4 degrees of freedom. We calculate the quantiles using our 1000 observations and compare it to the theoretical quantile. We repeat this experiment 1000 times to get a feeling of the variability of our quantile estimate from the observations.



```python
pv = [0.0001,0.001,0.01,0.05,0.2,0.5,0.8,0.95,0.99,0.999,0.9999]
simulate_quantile = np.zeros((len(pv), 1000))
theoretical_quantile = np.zeros((len(pv), 1000))

for i in range(1000):
    simulate_quantile[:,i] = np.quantile(t.rvs(df =4, size = 1000), pv)
    theoretical_quantile[:, i] = t.ppf(pv, df = 4)
    
```


```python

for j, _ in enumerate(pv):
    plt.plot(theoretical_quantile[j,:], simulate_quantile[j, :], color = 'black')

plt.plot([-20, 20],[-20, 20])
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Simulated Quantiles')
```



    
![png](extremes_34_1.png)
    


We see that the non-center simulated quantiles have much higher variability. This examples shows that the tail quantiles of the empirical distribution
are inaccurate predictors of the true quantiles. Thus, when calculating tail quantiles, such as VaR, one has to fit a distribution to the data, rather then estimating the quantiles from the observed data.

The usual approach is to use the empirical distribution in the main body of the data  between the 5% and 95% quantiles), while fitting a heavy tail distribution to the tails, for example the pareto distribution.

# Estimating the Tail losses


```python
q05
```




    -0.01545860011031408




```python
q05 = np.quantile(sp500['log_return'], 0.05)  # 5% quantile
losses = - sp500['log_return'].loc[sp500['log_return'] < q05] +q05
params =  expon.fit(losses, loc = 0, scale = 1)
params_t = t.fit(losses, loc = 0)
```


```python

```




    (1.6520658512964799, 0.005612857735874041, 0.004692617830430222)




```python
plt.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), empirical_cdf(sorted(sp500['log_return'].loc[sp500['log_return'] < q05])))
```




    
![png](extremes_40_1.png)
    




```python
expon_tail_cdf = 0.05*np.sort((1-expon.cdf(losses, loc = params[0], scale = params[1])))
t_tail_cdf = 0.05*np.sort(1-t.cdf(losses,df = params_t[0], loc = params_t[1], scale = params_t[2]))


fig, ax = plt.subplots(1,1, figsize = (15,5))
ax.plot(sorted(sp500['log_return']), empirical_cdf(sorted(sp500['log_return'])), label = 'empirical')
ax.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), expon_tail_cdf, label = 'expon_tail_cdf')
ax.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), t_tail_cdf, label = 't_tail_cdf')
ax.legend()
ax.set_ylim([0.0, 0.05])
ax.set_xlim([-0.15, 0.0])
```



    
![png](extremes_42_1.png)
    


# Threshold exceedances

What is the distribution of \\(X(t)\\) given \\(X(t)\\) exceeds some high threshold (for example, its 0.99 quantile)? The generalized Pareto Distribution has been shown to model the tail distribution given that the threshold is large enough GIVEN that the maximum can be modelled by a General extreme value distribution, which most distributions considered have.


```python
q05 = np.quantile(sp500['log_return'], 0.05)  # 5% quantile
losses = - sp500['log_return'].loc[sp500['log_return'] < q05] +q05
params_genpareto = genpareto.fit(losses, loc = 0)
params_genpareto
```




    (0.43767003369774377, 3.0926415630468514e-05, 0.0067057240009978335)




```python
genpareto_tail_pdf = genpareto.pdf(sorted(losses), params_genpareto[0], params_genpareto[1], params_genpareto[2])
genpareto_tail_cdf = genpareto.cdf(sorted(losses), params_genpareto[0], params_genpareto[1], params_genpareto[2])
genpareto_tail_cdf = 0.05*np.sort(1-genpareto_tail_cdf)

fig, ax = plt.subplots(1,1, figsize = (15,5))
ax.hist(losses, density=True, bins = 80)
ax.plot(sorted(losses), genpareto_tail_pdf)
```



    
![png](extremes_47_1.png)
    



```python
fig, ax = plt.subplots(1,1, figsize = (15,5))
ax.plot(sorted(sp500['log_return']), empirical_cdf(sorted(sp500['log_return'])), label = 'empirical')
ax.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), genpareto_tail_cdf, label = 'expon_tail_cdf')
ax.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), t_tail_cdf, label = 't_tail_cdf')
ax.plot(sorted(sp500['log_return'].loc[sp500['log_return'] < q05]), genpareto_tail_cdf, label = 'genpareto_cdf')
ax.legend()
ax.set_ylim([0.0, 0.05])
ax.set_xlim([-0.15, 0.0])
```


    
![png](extremes_48_1.png)
    


# Copulas

Modelling both indexes. First we model the volatility to get the residuals. Then we apply a copula to the residuals, as the residuals is the loss.

The Garch(1,1) model is for returns \\\(r_t\\) is:

$$
\begin{aligned}
r_t &= \mu_t + \epsilon_t \\\\\
\epsilon_t &=z_t\sigma_t \\\\\
\sigma_t^2 &= \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1\sigma_{t-1}^2
\end{aligned}
$$
 \\(\alpha_0, \alpha_1, \beta_1 > 0\\) as volatility is positive. \\(z_t\\) is iid and \\(E[z_t] = 0\\) and \\(Var[z_t] = 1\\). \\(mu_t\\) is usually assumed to be constant. Let \\(u_t = (\epsilon_t^2 - sigma_t^2 )\\) and plug it in then

$$ \epsilon_t^2 =\alpha_0 + (\alpha_1 + \beta_1) \epsilon_{t-1}^2 +  u_t -\beta_1u_{t-1}$$

We see that this is simply an ARIMA process. We set the constraints \\(0 < \alpha_1 + \beta_1  <1\\) to get a mean-reverting process.

Also, More realistic VaR estimation with GARCH -> VaR = mean + (GARCH vol) * quantile


Let's fit a Garch model.


```python
from arch import arch_model
sp500_garch_model = arch_model(sp500['log_return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
sp500_garch_fit = sp500_garch_model.fit()

nasdaq_garch_model = arch_model(nasdaq['log_return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
nasdaq_garch_fit = nasdaq_garch_model.fit()

plt.plot(sp500_garch_fit.resid)
```





    
![png](extremes_52_3.png)
    



```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Perform the Ljung-Box test, do
# The null hypothesis of Ljung-Box test is: the data is independently distributed.
lb_test = acorr_ljungbox(sp500_garch_fit.resid , lags = 10)

# Store p-values in DataFrame
df = pd.DataFrame({'P-values': lb_test.iloc[:,1]}).T

# Create column names for each lag
col_num = df.shape[1]
col_names = ['lag_'+str(num) for num in list(range(1,col_num+1,1))]

# Display the p-values
df.columns = col_names
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lag_1</th>
      <th>lag_2</th>
      <th>lag_3</th>
      <th>lag_4</th>
      <th>lag_5</th>
      <th>lag_6</th>
      <th>lag_7</th>
      <th>lag_8</th>
      <th>lag_9</th>
      <th>lag_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>P-values</th>
      <td>2.103113e-16</td>
      <td>4.272782e-21</td>
      <td>2.473644e-20</td>
      <td>7.223487e-23</td>
      <td>1.702407e-23</td>
      <td>6.703007e-31</td>
      <td>6.231727e-45</td>
      <td>1.095580e-54</td>
      <td>4.656047e-62</td>
      <td>2.200335e-62</td>
    </tr>
  </tbody>
</table>
</div>




```python
nasdaq_garch_fit.resid
```




    1      -0.020776
    2       0.009954
    3       0.033189
    4      -0.000138
    5      -0.025576
              ...   
    2511    0.006905
    2512    0.002842
    2513    0.006839
    2514    0.003130
    2515    0.002464
    Name: resid, Length: 2515, dtype: float64



## Copula fit

First fit the marginals


```python
sp500_nct_params = nct.fit(sp500_garch_fit.resid)
sp500_nct_fit = nct.pdf(sorted(sp500['log_return']), df = sp500_nct_params[0], nc = sp500_nct_params[1], loc = sp500_nct_params[2], scale = sp500_nct_params[3])

nasdaq_nct_params = nct.fit(nasdaq_garch_fit.resid)
nasdaq_nct_fit = nct.pdf(sorted(nasdaq['log_return']), df = nasdaq_nct_params[0], nc = nasdaq_nct_params[1], loc = nasdaq_nct_params[2], scale = nasdaq_nct_params[3])
```

Generate Uniforms


```python
sp500_uni = nct.cdf(sp500_garch_fit.resid, df = sp500_nct_params[0], nc = sp500_nct_params[1], loc = sp500_nct_params[2], scale = sp500_nct_params[3] )
nasdaq_uni = nct.cdf(nasdaq_garch_fit.resid, df = nasdaq_nct_params[0], nc = nasdaq_nct_params[1], loc = nasdaq_nct_params[2], scale = nasdaq_nct_params[3] )
```


```python
plt.scatter(sp500_uni, nasdaq_uni)
```




    
![png](extremes_60_1.png)
    


The key point is that the scatterplots on the unit square \\([0; 1]\\) remove the influence of the marginal distributions. As a consequence they allow us to focus on the dependency structure between the two random variables Z1 (S&P500) and Z2 (Nasdaq).

To fit a a copula one can do a maximum likelihood estimation. Once can also calculate kendell's tau and estimate the parameter using the estimated kendell's tau as kendell's tau is known for many copulas.


```python
def clayton(u1, u2, theta):

    return np.power(np.power(u1, -theta) + np.power(u2, -theta) - 1, -1.0 / theta)
```

Calculate kendell tau to fit the clayton copula:


```python
tau, p_val = kendalltau(sp500_uni, nasdaq_uni)
```


```python
theta = 2 * tau / (1 - tau)
theta
```

## Simulating Copulas

We can simulate from paramteric copulas, using the following conditional probability:

$$ C_{u_1}(u_2) =  P( U_2 < u_2 | U_1 = u_1) = \lim_{h \to 0} \frac{C(u_1 + h,u_2) - C(u_1,u_2)}{h} = \frac{\partial C(u_1, u_2)}{\partial u_1}$$

The procedure is following:

Generate random variables \\(u_1\\) and \\(t\\) and set \\(u_2 =C_{u_1}^{-1}(t)\\). The desired dependent pair is \\((u_1, u_2)\\)

If we use the Gumbell Copula:

$$C(u,v) =\exp \Big( - \big( (-\ln u)^\alpha + (-\ln v)^\alpha \big)^{\frac{1}{\alpha}} \Big) $$ 

We get:

$$C_u(u_2) =  C(u_1,u_2) * \frac{(-\ln u_1)^{\alpha-1}}{u_1} \Big(  (-\ln u_1)^\alpha + (-\ln u_2)^\alpha \big)^{\frac{1}{\alpha}-1}\Big)$$

Computing the inverse is a bit hard, we need to do it numerically

Let's simulate them:



```python
def Gumbel(u,v, alpha):
    

    return np.exp( -(np.power(-np.log(u), alpha) + np.power(-np.log(v), alpha)  ) ** (1.0 / alpha))

def cond_gumbel(u,v, alpha):

    part1 = np.power(-np.log(u), alpha -1) / u
    part2 = ( np.power(-np.log(u), alpha) + np.power(-np.log(v), alpha)) ** ((1.0 / alpha) - 1.0)

    return Gumbel(u,v, alpha) * part1 * part2


def inverse_obj(v, t, u, alpha):
    """
    Solve C_u(v) = t given u and alpha
    """


    return cond_gumbel(v, u, alpha) - t


def inverse(t, u, alpha,):


    res = brentq(inverse_obj, 1e-16, 1, args = (t, u, alpha))

    return res

inverse_vec = np.vectorize(inverse, excluded= ('alpha', 'init_x'))


```


```python
u = np.array([99])
u
```




    array([99])




```python
u = np.array([99])
v = np.array([99])

for i in range(100):

    u_tmp = np.random.uniform(size = 10)
    t = np.random.uniform(size = 10)

    # Numerical erros
    try:
        v_tmp = inverse_vec(t, u_tmp, 4.0)
    except ValueError:
        v_tmp = None

    if not v_tmp is None:
        u = np.hstack((u, u_tmp))
        v = np.hstack((v, v_tmp))

```


```python
plt.scatter(u[1:], v[1:])
```





    
![png](extremes_74_1.png)
    

