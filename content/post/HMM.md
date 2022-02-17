+++
title = "Hidden Markov Models"
date = "2022-02-15"
author = "Ragnar Levi Gudmundarson"
tags = ["Model", "Classification"]
+++


In this notebook I will be exploring a Hidden Markov Model (HMM) to
classify bull and bear states of a simulated financial instrument.

``` r
library(depmixS4)
library(ggplot2)
library(matrixStats)
```

Simulate the data. Assume returns are normally distributed. Where the
mean and the standard deviation depends on the state of the financial
market. We will only consider 2 states.

``` r
set.seed(42)

Nk_lower <- 50
Nk_upper <- 150
bull_mean <- 0.1
bull_var <- 0.1
bear_mean <- -0.05
bear_var <- 0.2

# Create the list of durations (in days) for each regime
days <- replicate(5, sample(Nk_lower:Nk_upper, 1))

# Create the various bull and bear markets returns
market_bull_1 <- rnorm( days[1], bull_mean, bull_var)
market_bear_2 <- rnorm( days[2], bear_mean, bear_var)
market_bull_3 <- rnorm( days[3], bull_mean, bull_var)
market_bear_4 <- rnorm( days[4], bear_mean, bear_var)
market_bull_5 <- rnorm( days[5], bull_mean, bull_var)

# Create the list of true regime states and full returns list
true_regimes <- c( rep(1,days[1]), rep(2,days[2]), rep(1,days[3]), rep(2,days[4]), rep(1,days[5]))
returns <- c( market_bull_1, market_bear_2, market_bull_3, market_bear_4, market_bull_5)

plot(returns, type="l", xlab='', ylab="Returns")
```

![](unnamed-chunk-2-1.png) 

By visual inspection we can clearly see when the instrument is in a good
and bad shape. We will start by using the package `depmixS4` to find the
hidden states. Afterwards we will try to replicate the results with our
own code.

``` r
hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 2, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)
```

    ## converged at iteration 20 with logLik: 299.9928

``` r
# Output both the true regimes and the
# posterior probabilities of the regimes

post_probs <- posterior(hmmfit, type = 'viterbi')
layout(1:2)
plot(post_probs$state, type='s', main='True Regimes', xlab='', ylab='Regime')
matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='topright', c('Bull','Bear'), fill=1:2, bty='n')
```

![](unnamed-chunk-4-1.png)

The model does a good job. Now we will go through the maths of a HMM
model. Let $x_t$ denote the observed state aka the stock return and let $z_t$ denote the hidden state governing the dynamics of $x_t$. The forward filter algorithm allows us to compute the filtered
marginals $p(z_t | x_{1:t})$
. Note that we can write:


\begin{equation}
  \begin{split}
    p(z_t = j | x_{1:t}) = p(z_t = j | x_t, x_{1:t-1}) = \frac{p(x_t | z_t = j) p(z_t = j | x_{1:t-1})}{p(x_t | x_{1:t-1})}
  \end{split}
\end{equation}

where we used the fact that $x_t$ does not depend on $x_{1:t-1}$. Using this we can calculate $p(z_t = j | x_{1:t})$ recursively. Denote $\boldsymbol{\alpha}_t = [p(z_t = 1 | x_{1:t}),\dots, p(z_t = K | x_{1:t})]^T$ and $\boldsymbol{Z}$ the transition matrix were $Z_ij$ denotes jump from $i$ to $j$
.
 $$\boldsymbol{\alpha}_t\propto\phi_t\odot  (\boldsymbol{Z}^T\boldsymbol{\alpha}_{t-1})   $$
</pre>
where $\boldsymbol{\phi}_t = [p(x_t | z_t = 1),\dots, p(x_t | z_t = K)]^T$ is a vector and $\boldsymbol{Z}$ is the transition matrix with entries $Z_{ij} = p(z_t =j | z_t = i)$
``` r
forward <- function(init_a, Z, phi, x){
  # init_a - initial distribution p(z_1 = j)
  # Z - transition matrix
  # phi list of probability densities p(x_t | z_t = j)


  normalization_const <- rep(0, length(x))
  a <- matrix(0, ncol = length(x), nrow = length(init_a))
  phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[1])


  a[, 1] <- phi_vec*init_a
  normalization_const[1] <- sum(a[,1])
  a[, 1] <- a[, 1]/sum(a[,1])
  a[is.nan(a[, 1]), 1] <- 0

  for(i in 2:length(x)){

    phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[i])

    a[,i] <-  phi_vec * (t(Z) %*% a[,i-1])
    normalization_const[i] <- sum(a[,i])

    a[, i] <- a[, i]/sum(a[,i])
    a[is.nan(a[,i]), i] <- 0

  }

  return(list(a = as.data.frame(t(a)), evidence = sum(log(normalization_const))))

}
```

Let’s test the filter on the data using the fitted parameters of hmmfit
(using hmmfit).

``` r
f_x1 <- function(x){dnorm(x, mean = getpars(hmmfit)[7], sd = getpars(hmmfit)[8])}
f_x2 <- function(x){dnorm(x, mean = getpars(hmmfit)[9], sd = getpars(hmmfit)[10])}



phi <- list(f_x1,f_x2)
Z <- matrix(getpars(hmmfit)[3:6], ncol = 2, byrow = TRUE)
init_a <- c(0,1)
filter_out <- forward(init_a, Z, phi, returns)

filter_out$a$days <- 1:length(returns)
filter_out$a$state <- factor(ifelse(filter_out$a$V1 > 0.5, "Bull", "Bear"))

ggplot(filter_out$a) + geom_line(aes(x = days, y = V1, color = "Bear")) +
  geom_line(aes(x = days, y = V2, color = "Bull"))+
  theme_minimal() +
  scale_color_manual(name='State',
                     breaks=c('Bear', 'Bull'),
                     values=c('Bear'='red', 'Bull'='darkgreen'))
```

![](unnamed-chunk-6-1.png)

The graph looks good, but it is a bit wiggly. We can produce smoother
lines if we allow for a non-online estimation, that is if we estimate
the state using all of our data, $p(z_t = j| x_{1:T})$
. This is simply called smoothing. We can write
 $$p(z_t = j| x_{1:T})\propto p(z_t = j| x_{1:t})p(x_{t+1:T}| z_t = j)$$
</pre>
Define $\boldsymbol{\beta}_t = [p(x_{t+1:T}| z_t = 1),\dots, p(x_{t+1:T}| z_t = K)]^T$
then we have
 $$p(z_t = j| x_{1:T})\propto\alpha_t(j)\beta_t(j)$$
</pre>
We can calculate $\beta_t(j)$
recursively as follows:

$$
\begin{equation}
  \begin{split}
    \beta_{t-1}(j) &= p(x_{t:T}| z_{t-1} = j ) \\\\\\ 
    &= \sum_i p(x_{t+1:T}, x_t, z_t = i| z_{t-1} = j ) \\\\\\  
    &= \sum_i p(x_{t+1:T} ,| z_t = i) p(x_t | z_t = i) p( z_t = i| z_{t-1} = j )
  \end{split}
\end{equation}
$$



We can write this in a matrix form
 $$\boldsymbol{\beta_{t-1}} =\boldsymbol{Z} (\boldsymbol{\phi}_t\odot\boldsymbol{\beta}_t) $$
</pre>
The smoother starts by calculating $\boldsymbol{\alpha}_t$ using the (forward) filter, we already discussed, and then uses the
(backwards) recursion to calculate $\boldsymbol{\beta}_t$
just described, hence the name of this algorithm is the
forward-backwards algorithm. We will additionally calculate the smoothed
transitions as we will need it later.

$$
\begin{equation}
  \begin{split}
    p(z_t = i, z_{t+1} = j | x_{1:T}) &\propto p(z_t | x_{1:t})p(z_{t+1} |z_t, x_{t+1:T}) \\\\\\ 
    &\propto p(z_t | x_{1:t})p(x_{t+1:T} | z_{t+1}, z_t)p(z_{t+1}| z_t) \\\\\\ 
    &\propto p(z_t | x_{1:t})p(x_{t+1} | z_{t+1})p(x_{t+2:T}|z_{t+1})p(z_{t+1}| z_t) \\\\\\ 
    &\propto p(z_t | x_{1:t})p(x_{t+1} | z_{t+1})p(x_{t+2:T}|z_{t+1})p(z_{t+1}| z_t) \\\\\\ 
    &= \alpha_t(i) \phi_{t+1}(j)\beta_{t+1}(j) Z_{ij}
  \end{split}
\end{equation}
$$

``` r
ForwardBackward <- function(init_a, Z, phi, x){

  T <- length(x)
  K <- length(init_a)

  filter_out <- forward(init_a, Z, phi, returns)
  a <- t(as.matrix(filter_out$a))

  b <- matrix(0, ncol = length(x), nrow = K)
  O <- array(1, dim=c(T-1,K,K))

  posterior <- matrix(0, ncol = length(x), nrow = K)

  phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[T])
  b[,T] <- (Z %*% phi_vec)
  posterior[,T] <- pmax(b[,T], c(1e-15,1e-15))
  posterior[,T] <- posterior[,T]/sum(posterior[,T])
  posterior[is.nan(posterior[, T]), T] <- 0



  for(t in T:2){

    phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[t-1])

    b[,t-1] <- Z %*% (phi_vec*b[,t])
    b[,t-1] <- pmax(b[,t-1], c(1e-15,1e-15))
    posterior[,t-1] <- a[, t-1]*b[,t-1]
    posterior[,t-1] <- posterior[,t-1]/sum(posterior[,t-1])
    posterior[is.nan(posterior[, t-1]), t-1] <- 0

    for(i in 1:K){
      for(j in 1:K){
        O[t-1,i,j] <- a[i, t-1]*phi_vec[j]*b[j,t]*Z[i,j]
      }
    }

    O[t-1,,] <- O[t-1,,]/sum(O[t-1,,])


  }


  return(list(posterior = as.data.frame(t(posterior)),
              gamma = posterior,
              O = O,
              a = a
))

}
```

``` r
f_x1 <- function(x){dnorm(x, mean = getpars(hmmfit)[7], sd = getpars(hmmfit)[8])}
f_x2 <- function(x){dnorm(x, mean = getpars(hmmfit)[9], sd = getpars(hmmfit)[10])}

phi <- list(f_x1,f_x2)
Z <- matrix(getpars(hmmfit)[3:6], ncol = 2, byrow = TRUE)
init_a <- c(0,1)


fb_out <- ForwardBackward(init_a, Z, phi, returns)$posterior

fb_out$days <- 1:length(returns)
fb_out$state <- factor(ifelse(fb_out$V1 > 0.5, "Bull", "Bear"))

ggplot(fb_out) + geom_line(aes(x = days, y = V1, color = "Bear")) +
  geom_line(aes(x = days, y = V2, color = "Bull"))+
  theme_minimal() +
  scale_color_manual(name='State',
                     breaks=c('Bear', 'Bull'),
                     values=c('Bear'='red', 'Bull'='darkgreen'))
```

![](unnamed-chunk-8-1.png)

By estimating the posterior using all observations via smoothing we can
see that we get more confident in our estimation. Instead of using the
forward-backward algorithm we can compute the most probable sequence of
states using the viterbi algorithm:
 $$\boldsymbol{z}^{*} =\arg\max_{z_{1:T}} p(z_{1:T} | x_{1:T}) $$
</pre>

Note that the jointly most probable sequence of states (Viterbi) is not
necessarily the same as the sequence of marginally most probable states
(if we maximize forward-backwards).
 $$\arg\max_{z_{1:T}} p(z_{1:T} | x_{1:T})\neq\Big(\arg\max_{z_1} p(z_1 | x_{1:T}),\dots, arg\max_{z_T} p(z_T | x_{1:T})\Big)$$
</pre>
let’ define $m_t(j) =\max_{z_1,\dots, z_{t-1}} p(z_{1:t-1}, z_t = j | x_{1:t})$ and Note that we represent the most probable path by taking the maximum
over allpossible previous state sequences. Given that we had already
computed the probability of being in every state at time $t-1$
, we compute the Viterbi probability by taking the most probable of the
extensions of the paths that lead to the current cell:
 $$m_t(j) =\max_{z_{t-1} = i} m_{t-1}(i) Z_{ij} p(x_t | z_t =j) $$
</pre>

``` r
viterbi <- function(init_a, Z, phi, x){

  best_state <- matrix(0, ncol = length(x), nrow = length(init_a))

  delta <- matrix(0, ncol = length(x), nrow = length(init_a))

  phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[1])

  delta[,1] <- init_a*phi_vec
  delta[,1] <- delta[,1]/sum(delta[,1])
  delta[is.nan(delta[,1]),1] <- 0
  best_state[1] <- which.max(init_a)

  # Calculate most probable path to j
  for(i in 2:length(x)){

    phi_vec <- sapply(phi, function(f, x){ f(x)}, x = x[i])
    delta[,i] <- sapply(1:length(init_a), function(x){max(delta[,i-1]*Z[,x]*phi_vec[x])})
    delta[,i] <- delta[,i]/sum(delta[,i])
    delta[is.nan(delta[,1]),1] <- 0
    best_state[,i] <- sapply(1:length(init_a), function(x){which.max(delta[,i-1]*Z[,x]*phi_vec[x])})

  }

  # Compute most probable state using traceback.

  z <- matrix(0, ncol = length(x), nrow = 1)
  s <- matrix(0, ncol = length(x), nrow = 1)

  z[length(x)] <- which.max(delta[, length(x)])

  for(i in length(x):2){
    z[i-1] <- best_state[z[i], i]

  }

  return(list(delta = delta, z = z, delta_df = as.data.frame(t(delta))))
}



out_viterbi <- viterbi(init_a, Z, phi, returns)

states <- as.data.frame(t(out_viterbi$z))
states$time <- 1:nrow(states)
states$State <- ifelse(states$V1 == 2, "Bull", "Bear")
ggplot(states) + geom_point(aes(x = time, y = V1, color = State)) +
    scale_color_manual(values=c('Bear'='red', 'Bull'='darkgreen')) +
  theme_minimal()
```

![](unnamed-chunk-9-1.png)

``` r
out_viterbi$delta_df$days <- 1:nrow(states)
ggplot(out_viterbi$delta_df) + geom_line(aes(x = days, y = V1, color = "Bear")) +
  geom_line(aes(x = days, y = V2, color = "Bull"))+
  theme_minimal() +
  scale_color_manual(name='State',
                     breaks=c('Bear', 'Bull'),
                     values=c('Bear'='red', 'Bull'='darkgreen'))
```

![](unnamed-chunk-9-2.png)

Finally, let’s write a function that estimates the HMM parameters and
compare it to the values of the `depmixs4` package. The Baum-Welch
algorithm, an Em algorithm for HMMs, is usually used. We are interested
in finding the parameters $\theta = (\mu_1,\mu_2,\sigma_1,\sigma_2,\boldsymbol{Z},\pi)$ where $\mu_k,\sigma_k$ is the mean and standard deviation of the returns in state $k$. To make it a bit more generalized, let $K$ be the number of states. In the E-step we calculate the expected value
using $p(\boldsymbol{z}|\boldsymbol{x},\theta)$ 

$$
\begin{aligned}
    Q(\theta,\theta^{old}) &= E_{\boldsymbol{z}|\boldsymbol{x},\theta^{old}}[\log p(\boldsymbol{z},\boldsymbol{x} |\theta)]\\\\\\
    &= E_{\boldsymbol{z}|\boldsymbol{x},\theta^{old}}[\log p(z_1 |\theta)\prod_{t = 2}^T\log p(z_t | z_{t-1},\theta)\prod_{t = 1}^T\log p(x_t | z_{t},\theta)]\\\\\\
    &=\sum_{i = 1}^K p(z_1 = i |\boldsymbol{x},\theta^{old})\log\pi_i\\\\\\
    & +\sum_{t = 1}^T\sum_{i = 1}^K\sum_{j = 1}^K p(z_{t-1} = j,z_t = i|\boldsymbol{x},\theta^{old})\log p(Z_{ji} |\theta)\\\\\\
    & +\sum_{t = 1}^T\sum_{i = 1}^K p(z_{t} = i|\boldsymbol{x},\theta^{old})\log p(x_t | z_{t},\theta)]\\\\\\
\end{aligned}
$$

Next we perform the M-step and we maximize $Q$ with respect to theta. The constraints are $\sum_i^K Z_{ji} = 1$ for all $j = 1,\dots, K$ and $\sum_i\pi_i = 1$
. The Lagrangian of the objective is:

$$
\begin{aligned}
    \arg\max_\theta Q(\theta) + \lambda(\sum_i \pi_i - 1) + \eta_j (\sum_i^K Z_{ji} -1)
\end{aligned}
$$

Taking derivative w.r.t to $\pi_i = p(z_1 = i |\boldsymbol{x},\theta^{old})/\lambda$, and using the fact that $\sum_i pi_1 =1$ we get $\lambda = 1$, with similar argument we get that 
$Z_{ji} = \frac{N_{ji}}{\sum_k N_{jk}}$ where $N_{ji} =\sum_{t = 2}^T p(z_{t-1} = j,z_t = i|\boldsymbol{x},\theta^{old})$. Because in this case we are assuming that $p(x_t | z_t,\theta)$ is a normal distribution we get:

$$
\begin{equation}
\mu_k =\frac{\sum_t^T\gamma_t(k) x_t}{\sum_t^T\gamma_t(k)},\quad,\sigma^2 =\frac{\sum_t^T\gamma_t(k)x_t^2 - N_j\mu^2}{N_j}
\end{equation}
$$

where $N_j =\sum_t^T p(z_{t} = j|\boldsymbol{x},\theta^{old})$
. With the equations at hand we write a function for the EM.

``` r
EM <- function(x, k, nr_itr = 200){
  # x - data
  # k - number of states
  # nr_itr - number of EM iterations

  Z <- matrix(rep(1,k^2)/k, nrow = k, ncol = k)

  means <- rnorm(k)
  sds <- c(2,2)
  init_a <- rep(1, k)/k




  for(n in 1:nr_itr){
    density <- lapply(1:k, function(i){function(x){dnorm(x, mean = means[i], sd = sds[i])}})
    fb_out <- ForwardBackward(init_a, Z, density, x)

    for(i in 1:k){
      for(j in 1:k){
        Z[i,j] <- sum(fb_out$O[,i,j])/sum(fb_out$O[,i,])
      }
    }

    for(i in 1:k){

      init_a[i] <- fb_out$gamma[i,1]

    }

    for(i in 1:k){

      means[i] <- sum(fb_out$gamma[i,]*x)/sum(fb_out$gamma[i,])

      sds[i] <- sqrt((sum(fb_out$gamma[i,]*x^2) - sum(fb_out$gamma[i,])*means[i]^2)/ sum(fb_out$gamma[i,]))

    }


  }

  return(list(means = means, sds = sds, Z = Z, init_a = init_a, fb_out = fb_out))

}

out_em <- EM(returns, 2)
print(out_em$means)
```

    ## [1]  0.09608514 -0.08617614

``` r
print(out_em$sds)
```

    ## [1] 0.1015317 0.2170290

``` r
print(out_em$Z)
```

    ##            [,1]        [,2]
    ## [1,] 0.99597796 0.004022039
    ## [2,] 0.00416305 0.995836950

Comparing to the depmaxs4 package

``` r
getpars(hmmfit)
```

    ##          pr1          pr2                                                      (Intercept)           sd  (Intercept)           sd
    ##  0.000000000  1.000000000  0.990073371  0.009926629  0.006200274  0.993799726 -0.084785623  0.217380580  0.094950502  0.103102669

We get very similar results, but both estimates are somewhat off the real parameters. Finally we can plot the posterior
probability of the states. The 

``` r
posterior <- out_em$fb_out$posterior
posterior$days <- 1:length(returns)
posterior$state <- factor(ifelse(posterior$V1 > 0.5, "Bull", "Bear"))

ggplot(posterior) + geom_line(aes(x = days, y = V1, color = "Bear")) +
  geom_line(aes(x = days, y = V2, color = "Bull"))+
  theme_minimal() +
  scale_color_manual(name='State',
                     breaks=c('Bear', 'Bull'),
                     values=c('Bear'='red', 'Bull'='darkgreen'))
```

![](unnamed-chunk-12-1.png)
