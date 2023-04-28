+++
title = "Generalized Linear Models"
date = "2022-03-02"
author = "Ragnar Levi Gudmundarson"
tags = ["Model"]
+++


In this notebook, I am going to note down elements of generalized linear
models (GLM). GLMs come from the exponential family where a response $y$
has the following distribution
 $$f(y;\theta) =\exp\Big(\frac{y\theta - b(\theta)}{a(\phi)} + c(y,\theta)\Big)$$
</pre>
where $\theta$ is the canonical parameter and $\phi$
is the dispersion parameter. Before continuing we define the score and
derive the a formula for the expected score and the information.

The score is the derivative of the log-likelihood and we can derive the
expected score as, using $f =  f(y;\theta)$
:

$$
\begin{equation}
  \begin{split}
E[\frac{\partial}{\partial\theta}\log f] &= E[\frac{1}{f}\frac{\partial}{\partial\theta} f]\\\\\\ 
&=\int\frac{1}{f}\frac{\partial f}{\partial\theta} fdx\\\\\\ 
&=\frac{\partial }{\partial\theta}\int  fdx\\\\\\ 
&=\frac{\partial }{\partial\theta} 1 = 0
  \end{split}
\end{equation}
$$

This formula gives us the mean function $E[y] =\mu = b'(\theta)$. The information equality is:

$$
  \begin{split}
E\Big[\frac{\partial^2}{\partial\theta\partial\theta^T}\log f\Big] &= E\Big[\frac{\partial}{\partial\theta} (\frac{1}{f}\frac{\partial f}{\partial\theta^T})\Big]\\\\\\ 
&= -E\Big[\frac{1}{f^2}\frac{\partial f}{\partial\theta}\frac{\partial f}{\partial\theta^T}\Big] + E\Big[\frac{1}{f}\frac{\partial^2 f}{\partial\theta\partial\theta^T}\Big]\\\\\\ 
&=  -E\Big[\frac{1}{f^2}\frac{\partial f}{\partial\theta}\frac{\partial f}{\partial\theta^T}\Big]\\\\\\ 
&= - E\Big[ (\frac{\partial\log f}{\partial\theta})(\frac{\partial\log f}{\partial\theta})^T\Big]
  \end{split}
$$

Which can be seen as $Var(\frac{\partial}{\partial\theta}\log f) = - E\Big[\frac{\partial^2}{\partial\theta\partial\theta^T}\log f\Big]$. The information equality gives the variance function $Var(y) = a(\phi)b''(\theta)$. GLMs make use of a linear predictor $\eta = x^T\beta$ where $x$ is a vector of features and $\beta$ the regression parameters to be estimated. The third part of GLMs is the
link function, which connects $\eta$ to $\mu$ via a function $g$, $g(\mu) =\eta$.

A typical example is the binomial response model $y\sim Bin(n,\pi)$
gives:
 $$\log f =\frac{y\log(\frac{\pi}{1-\pi}) +\log(1-\pi)}{1} + const$$
</pre>
The canonical parameter is $\theta =\log(\frac{\pi}{1-\pi})$ and $\phi = 1$ Another example is the Poisson response model, $y\sim Poi(\lambda)$
which gives
 $$\log f =\frac{y\log(\lambda) -\lambda}{1} + const$$
</pre>
Here the canonical parameter is $\theta =\log(\lambda)$ and $\phi = 1$ When $\eta =\theta$
we say that the model has a canonical link. It is often convenient to
use a canonical link but that does not mean that the data is actually
generated from that model. This has to be checked using some
diagnostics.

When we fit GLM models we calculate need to calculate the derivative of
the log-likelihood $l$</pre> $$\frac{\partial l}{\partial\beta_i} =\frac{\partial l}{\partial\theta}\frac{\partial\theta}{\partial\mu}\frac{\partial\mu}{\partial\eta}\frac{\partial\eta }{\partial\beta_i}$$
</pre>

We have:
 $$\frac{\partial l}{\partial\theta} =\frac{y-b'(\theta)}{a(\phi)} =\frac{y-\mu}{a(\phi)}$$
</pre>
using the fact that $\mu = b'(\theta)$ and that $f'(x) =\frac{1}{(f^{-1})'(y)}$
we have:
 $$\frac{\partial\theta}{\partial\mu} =\frac{1}{b''(\theta)} =\frac{a(\phi)}{Var(y)}$$
</pre> $\frac{\partial\mu}{\partial\eta}$ will depend on the link and $\frac{\partial\eta }{\partial\beta_i} = x_{ji}$. If we use a canonical link we have $\frac{\partial\mu}{\partial\eta} =\frac{\partial\mu}{\partial\theta} = b''(\theta)$
. Also if we use a canonical link then the Fisher scoring optimization
procedure is the same thing as Newton-Raphson optimization.

The dispersion parameter is usually fitted separately using a
traditional estimate of the variance.

Finally to comment on diagnostics. We can inspect the residuals, either
the Pearson residual is defined as
 $$r_i =\frac{y_i -\widehat{\mu}}{\sqrt{\widehat{Var}(y)}}=\frac{y_i -\widehat{\mu}}{\sqrt{a(\hat{\phi})V(\hat{\mu})}}$$
</pre>

or the deviance residual

$r_i = sign(y-\\mu)\\sqrt{d}\_i$

where $d_i$ is the deviance of observation $i$. If we plot the $\hat{\eta}$ as a function of the residuals, we should see a horizontal band with
a mean of approximately zero, something similar to white noise. If there is
a curvature in the plot then we might have a wrong link function or we
are missing some nonlinear feature, for example, $x^2$. If the horizontal band is not symmetric, not symmetric, or many outliers then the variance function may be incorrect, for example too low or high dispersion.

If we plot individual features vs residuals vs we should also get a
horizontal band. A curvature suggests that the feature should be
nonlinear.

If we plot the fitted values vs absolute residuals and do not see any
trend, then the variance function is ok. However, if there is a trend
then the variance function is incorrect. However, if we are in a certain
parametric family then we can not change the variance function.
However, we could perform a quasi-likelihood approach.

The main idea of the quasi-likelihood is to relax the exponential family
assumption and allow a more general variance component $\phi V(\mu)$ to define a quasi-score $\tilde{S} =\frac{y-\mu}{\phi V(\mu)}$. The quasi-score is similar to the score, namely, it satisfies the
expected score and information formulas $E[\tilde{S}]$ and $Var(\tilde{S}) =\frac{1}{\phi V(\mu)}$. As the score is the derivative of the log-likelihood, the quasi
log-likelihood $Q$ is the integral of the quasi score:
 $$Q(y;\mu) =\int_{y}^\mu\frac{y-t}{\phi V(t)} dt $$
</pre>

