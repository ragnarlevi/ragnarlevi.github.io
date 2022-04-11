+++
title = "Neural Networks for Insurance Pricing"
date = "2022-04-11"
author = "Ragnar Levi Gudmundarson"
tags = ["Model"]
+++

In this notebook we will be looking at a typicall insurance pricing data set and test GLM and NN models. A typical assumption is that the repsone is poisson:
$$ P(Y = y) = \frac{\lambda^y \exp(-\lambda)}{y!}$$
To note a shortcoming of the Poisson distribution is that the mean is equal to the variance, and thus one might use quasi-Poisson or a negative binomial instead. Another shortcoming is that insurance claims are usally zero inflated and thus a zero-inflated poisson model or a hurdle poisson model might be used instead. We will not consider any of those models as this notebook is simply testing some NN's and compare them to a simple GLM.

In a Poisson regression, the canonical link function is given by the logarithm:


$$ g(\lambda) = \log \lambda = x^T\beta$$

which is same as saying that the probability is given by an exponential function

$$\lambda = e^{  x^T\beta}$$
This is what makes glms so interpretable, we can simply think of the covariates as a multiplication factors with different weights.




Given a sample $(x_i, y_i)_{i=1}^n$, the log-likelihood is:

$$l(\beta) = \sum_i \Big(y_i log\lambda - \lambda  - \log y! \Big)  $$

or 
$$l(\beta) = \sum_i \Big(y_i x^T\beta  - e^{x^T\beta} - \log y! \Big) $$


The $l$ is differentible so it is possible to train the model by some gradient descent methods. Once a model has been fit we can assess the goodness-of-fit of a model is to compare its log-likelihood with that of a saturated model. The relevant test statistic, called the deviance, is defined by:

$$D = 2 \Big(l(\beta_{max}) - l(\hat{\beta}) \Big) $$

where $\beta_{max}$ is obtained by having one parameter for each data point, resulting in $\hat{\pi}_i = y_i$ and a perfect fit. $\hat{\beta}$ are the estimate parameters of our model.


The idea of Neural networks is to estimate the function $f:x \mapsto y$ by using multiple function layers. If there are $K$ layers then we define the $k$ layer function $f^{(k)}: R^{q_{k-1}} \mapsto R^{q_{k}}$ using $f^{(k)}(z^{(k-1)}) = \phi_k(Wz^{(k-1)})$ where $W$ is a matrix, $q_k$ is the dimension of hidden layer $k$ and $\phi_k$ is called an activation function. Note that $f^{(k)}$ returns a vector. We can also write this as

$$f^{(k)}: z^{k-1} \mapsto z^{k}: f^{(k)}(z^{(k-1)}) = \Big[ \phi_{k_1}([Wz^{(k-1)}]_{1}), \dots, \phi_{q_k} ([Wz^{(k-1)}]_{q_k})  \Big]^T$$
 
There are multiple choices for $\phi_k$ such as the sigmoid, Which is not advisable to be used as the activation function for several layers as it is likely to encounter the problem of vanishing  gradients, Hyperbolic functions: which  might also lead to the vanishing gradient problem, thus it is recommended to use it only for layers close to the output layer, ReLu and eLu. [ML cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu)


Note that $z^{(0)} = x $ and $z^{(K)} = y $ so we can see that if we set $K=1$ and use a exponential activation we simply get the Poisson regression.


Let's start by comparing Poisson regression and NN single layer regression. 


``` r
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.5     v purrr   0.3.4
    ## v tibble  3.1.6     v dplyr   1.0.8
    ## v tidyr   1.2.0     v stringr 1.4.0
    ## v readr   2.1.2     v forcats 0.5.1

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(keras)
library(tfdatasets)
library(stats)
library(MASS)
```

    ## 
    ## Attaching package: 'MASS'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

I will preprocess the data by scaling the continuous features. This is
mainly important for the NN as otherwise we might have a vanishing
gradient problem

``` r
set.seed(121)
mtpl <- read.csv("freMTPL2freq.csv")

PreProcess.Continuous <- function(data) 2*(data-min(data))/(max(data)-min(data))-1


mtpl$VehPower <- PreProcess.Continuous(mtpl$VehPower)
mtpl$Density <- PreProcess.Continuous(round(log(mtpl$Density),2))
mtpl$DrivAge <- PreProcess.Continuous(pmin(mtpl$DrivAge,90))
mtpl$VehAge <- PreProcess.Continuous(pmin(mtpl$VehAge,20))
mtpl$ClaimNb <- pmin(mtpl$ClaimNb, 5)  # truncate abnormal claim counts
mtpl$Exposure <- pmin(mtpl$Exposure, 1)

mtpl$Area <- as.factor(mtpl$Area)
mtpl$VehBrand <- as.factor(mtpl$VehBrand)
mtpl$Region <- as.factor(mtpl$Region)
# The bonus malus is a bit weird, multiple classes with very few observatins. We remove them
mtpl$BonusMalus <- as.factor(pmin(mtpl$BonusMalus, 150))
mtpl <- mtpl[mtpl$BonusMalus %in% names(table(mtpl$BonusMalus))[table(mtpl$BonusMalus)>10], ]


ggplot(mtpl)+geom_histogram(aes(ClaimNb))+theme_minimal()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](NNfun_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
mtpl$claim <- pmin(mtpl$ClaimNb, 1)
train_ind <- sample(1:nrow(mtpl), 0.7*nrow(mtpl))
mtpl_train <- mtpl[train_ind,]
mtpl_test <- mtpl[-train_ind, ]


X_train <- model.matrix(~-1+ ., mtpl_train[, c("Area", "VehPower", "VehAge" , "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region")])
X_test <- model.matrix(~-1+ ., mtpl_test[, c("Area", "VehPower", "VehAge" ,"DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region")])
```

Fit a one layer NN equal to a Poisson regression:

``` r
d <- dim(X_train)
model_lr <- keras_model_sequential()
```

    ## Loaded Tensorflow version 2.8.0

``` r
  model_lr %>%layer_dense(units = 1, activation = "exponential", input_shape = ncol(X_train), use_bias = TRUE)
  
  model_lr %>% compile(loss = 'poisson', optimizer = "nadam")
  
  model_lr %>% fit(x = X_train , 
                   y = mtpl_train$ClaimNb,
                   epochs = 500,
                   batch_size = 10000 ,
                   validation_split = 0.2,
                   callbacks = list(callback_early_stopping(monitor = "val_loss", patience =5)))
```

We can extract the weights.

``` r
keras::get_weights(model_lr)
```

    ## [[1]]
    ##                [,1]
    ##   [1,] -1.603317380
    ##   [2,] -1.538048029
    ##   [3,] -1.469263434
    ##   [4,] -1.329705119
    ##   [5,] -1.279911995
    ##   [6,] -1.209258556
    ##   [7,]  0.003041624
    ##   [8,] -0.290724725
    ##   [9,]  0.294765741
    ##  [10,] -0.186617434
    ##  [11,]  0.354943722
    ##  [12,]  0.354688674
    ##  [13,] -0.144094139
    ##  [14,]  0.449503154
    ##  [15,]  0.612444341
    ##  [16,] -0.165575519
    ##  [17,]  0.860628068
    ##  [18,]  0.391107976
    ##  [19,]  0.100037389
    ##  [20,]  0.443079144
    ##  [21,]  1.207042336
    ##  [22,]  0.723411620
    ##  [23,] -0.102506630
    ##  [24,]  0.673325658
    ##  [25,]  0.531370521
    ##  [26,]  0.706401825
    ##  [27,] -0.040133797
    ##  [28,]  0.620303273
    ##  [29,]  0.484823078
    ##  [30,]  0.849467278
    ##  [31,]  0.083374299
    ##  [32,]  1.010002255
    ##  [33,]  0.574869037
    ##  [34,]  1.076051235
    ##  [35,]  0.121795148
    ##  [36,]  1.417718768
    ##  [37,]  1.104706526
    ##  [38,]  0.547128618
    ##  [39,]  0.253262371
    ##  [40,]  0.603815496
    ##  [41,]  1.020774603
    ##  [42,]  1.337180138
    ##  [43,]  1.172135711
    ##  [44,]  0.360801071
    ##  [45,]  0.982879877
    ##  [46,]  0.383328229
    ##  [47,]  1.290437341
    ##  [48,] -0.336998910
    ##  [49,]  0.369966328
    ##  [50,]  1.392116785
    ##  [51,]  0.493119419
    ##  [52,]  1.141632557
    ##  [53,] -0.201187059
    ##  [54,]  0.603137255
    ##  [55,]  1.504844546
    ##  [56,]  1.326750278
    ##  [57,]  0.186605886
    ##  [58,] -0.376424700
    ##  [59,]  0.810333490
    ##  [60,]  1.565439820
    ##  [61,]  2.147293329
    ##  [62,]  1.599472880
    ##  [63,] -2.686163902
    ##  [64,]  1.346087456
    ##  [65,]  1.244910240
    ##  [66,]  1.533233762
    ##  [67,]  1.570108533
    ##  [68,]  1.904701710
    ##  [69,]  1.709696174
    ##  [70,] -2.936032057
    ##  [71,]  1.387654901
    ##  [72,]  0.721305966
    ##  [73,]  0.667430878
    ##  [74,]  1.875365376
    ##  [75,]  0.902625799
    ##  [76,]  0.159701601
    ##  [77,]  1.481162548
    ##  [78,] -3.475115538
    ##  [79,]  1.316375017
    ##  [80,] -0.166711271
    ##  [81,]  0.155739918
    ##  [82,]  0.049401805
    ##  [83,]  0.015968367
    ##  [84,]  1.765787244
    ##  [85,]  0.509474695
    ##  [86,] -0.110909089
    ##  [87,]  0.033204675
    ##  [88,]  1.586613059
    ##  [89,]  1.043718815
    ##  [90,] -0.037447035
    ##  [91,]  0.054667398
    ##  [92,] -2.279501438
    ##  [93,]  1.351941705
    ##  [94,]  1.849181652
    ##  [95,]  0.035675362
    ##  [96,]  0.174874589
    ##  [97,] -0.204305485
    ##  [98,]  1.782384753
    ##  [99,]  1.747193933
    ## [100,] -0.209538370
    ## [101,]  1.844644427
    ## [102,] -0.049750902
    ## [103,] -0.021507265
    ## [104,] -0.119137540
    ## [105,] -0.011201694
    ## [106,] -0.291927069
    ## [107,] -0.012972042
    ## [108,] -0.045153335
    ## [109,] -0.085200489
    ## [110,]  0.003353591
    ## [111,] -0.109741919
    ## [112,]  0.112893291
    ## [113,] -0.197463557
    ## [114,] -0.008845681
    ## [115,]  0.021557970
    ## [116,] -0.407801747
    ## [117,]  0.117297582
    ## [118,]  0.125498757
    ## [119,] -0.072155401
    ## [120,] -0.303847909
    ## [121,] -0.153249592
    ## [122,]  0.022824710
    ## [123,] -0.554389954
    ## [124,] -0.038375080
    ## [125,]  0.210243285
    ## [126,] -0.062201113
    ## [127,] -0.193821877
    ## [128,] -0.326262385
    ## [129,]  0.156235218
    ## [130,]  0.049883407
    ## [131,] -0.453637421
    ## [132,] -0.211725429
    ## [133,] -0.164955288
    ## [134,]  0.024733299
    ## 
    ## [[2]]
    ## [1] -1.684777

Fit a Poisson model with same features

``` r
out <- glm(ClaimNb ~., data =  mtpl_train[, c("Area", "VehPower", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "ClaimNb")],
           family = poisson())

summary(out)
```

    ## 
    ## Call:
    ## glm(formula = ClaimNb ~ ., family = poisson(), data = mtpl_train[, 
    ##     c("Area", "VehPower", "DrivAge", "BonusMalus", "VehBrand", 
    ##         "VehGas", "Density", "Region", "ClaimNb")])
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.0479  -0.3326  -0.3044  -0.2782   6.1671  
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)    -3.210093   0.044536 -72.078  < 2e-16 ***
    ## AreaB           0.009258   0.030474   0.304 0.761283    
    ## AreaC           0.011697   0.039489   0.296 0.767068    
    ## AreaD           0.046957   0.059860   0.784 0.432777    
    ## AreaE           0.024790   0.079747   0.311 0.755911    
    ## AreaF           0.034683   0.109573   0.317 0.751604    
    ## VehPower       -0.017993   0.018095  -0.994 0.320045    
    ## DrivAge         0.287095   0.018736  15.323  < 2e-16 ***
    ## BonusMalus51   -0.198463   0.051964  -3.819 0.000134 ***
    ## BonusMalus52    0.403083   0.066287   6.081 1.20e-09 ***
    ## BonusMalus53    0.377070   0.082262   4.584 4.57e-06 ***
    ## BonusMalus54   -0.182660   0.049983  -3.654 0.000258 ***
    ## BonusMalus55    0.456466   0.060038   7.603 2.90e-14 ***
    ## BonusMalus56    0.546272   0.073617   7.420 1.17e-13 ***
    ## BonusMalus57   -0.172866   0.048894  -3.536 0.000407 ***
    ## BonusMalus58    0.870648   0.048289  18.030  < 2e-16 ***
    ## BonusMalus59    0.384869   0.090032   4.275 1.91e-05 ***
    ## BonusMalus60    0.085745   0.044511   1.926 0.054054 .  
    ## BonusMalus61    0.446808   0.109577   4.078 4.55e-05 ***
    ## BonusMalus62    1.223501   0.037163  32.922  < 2e-16 ***
    ## BonusMalus63    0.754593   0.070840  10.652  < 2e-16 ***
    ## BonusMalus64   -0.120825   0.048074  -2.513 0.011960 *  
    ## BonusMalus65    0.693522   0.104155   6.659 2.77e-11 ***
    ## BonusMalus66    0.591490   0.116733   5.067 4.04e-07 ***
    ## BonusMalus67    0.758558   0.075934   9.990  < 2e-16 ***
    ## BonusMalus68   -0.024287   0.045425  -0.535 0.592887    
    ## BonusMalus69    0.751162   0.115866   6.483 8.99e-11 ***
    ## BonusMalus70    0.696236   0.121731   5.719 1.07e-08 ***
    ## BonusMalus71    0.817906   0.084692   9.657  < 2e-16 ***
    ## BonusMalus72    0.090669   0.043474   2.086 0.037017 *  
    ## BonusMalus73    0.939221   0.114397   8.210  < 2e-16 ***
    ## BonusMalus74    0.631192   0.162602   3.882 0.000104 ***
    ## BonusMalus75    1.078858   0.093579  11.529  < 2e-16 ***
    ## BonusMalus76    0.101600   0.043797   2.320 0.020350 *  
    ## BonusMalus77    1.433624   0.087580  16.369  < 2e-16 ***
    ## BonusMalus78    1.140805   0.127439   8.952  < 2e-16 ***
    ## BonusMalus79    0.436495   0.288924   1.511 0.130849    
    ## BonusMalus80    0.226085   0.041846   5.403 6.56e-08 ***
    ## BonusMalus81    0.615267   0.179958   3.419 0.000629 ***
    ## BonusMalus82    1.057732   0.196417   5.385 7.24e-08 ***
    ## BonusMalus83    1.203514   0.149494   8.051 8.24e-16 ***
    ## BonusMalus84    1.148742   0.316463   3.630 0.000283 ***
    ## BonusMalus85    0.319899   0.041324   7.741 9.85e-15 ***
    ## BonusMalus86    0.887996   0.204417   4.344 1.40e-05 ***
    ## BonusMalus87    0.122732   0.408460   0.300 0.763815    
    ## BonusMalus88    1.312807   0.174446   7.526 5.25e-14 ***
    ## BonusMalus89    0.184082   0.707237   0.260 0.794646    
    ## BonusMalus90    0.346293   0.041211   8.403  < 2e-16 ***
    ## BonusMalus91    1.328106   0.218532   6.077 1.22e-09 ***
    ## BonusMalus92    0.302787   0.408415   0.741 0.458468    
    ## BonusMalus93    1.106053   0.229752   4.814 1.48e-06 ***
    ## BonusMalus94   -0.586803   1.000169  -0.587 0.557403    
    ## BonusMalus95    0.526171   0.038015  13.841  < 2e-16 ***
    ## BonusMalus96    1.454482   0.242873   5.989 2.12e-09 ***
    ## BonusMalus97    1.307871   0.267604   4.887 1.02e-06 ***
    ## BonusMalus98    0.862244   0.707446   1.219 0.222915    
    ## BonusMalus99    0.649554   0.577553   1.125 0.260731    
    ## BonusMalus100   0.772885   0.032539  23.753  < 2e-16 ***
    ## BonusMalus101   1.585609   0.250397   6.332 2.41e-10 ***
    ## BonusMalus102   2.187316   0.289003   7.568 3.78e-14 ***
    ## BonusMalus103   1.507607   0.447423   3.370 0.000753 ***
    ## BonusMalus104   1.777887   0.707428   2.513 0.011965 *  
    ## BonusMalus105   1.402667   0.577535   2.429 0.015153 *  
    ## BonusMalus106   1.196992   0.080719  14.829  < 2e-16 ***
    ## BonusMalus107   1.540399   0.408530   3.771 0.000163 ***
    ## BonusMalus108   1.675492   0.577783   2.900 0.003733 ** 
    ## BonusMalus109   1.918022   0.707418   2.711 0.006702 ** 
    ## BonusMalus110   1.797438   0.447495   4.017 5.90e-05 ***
    ## BonusMalus111  -9.150438  78.461128  -0.117 0.907158    
    ## BonusMalus112   1.382636   0.072952  18.953  < 2e-16 ***
    ## BonusMalus113   0.694298   0.707323   0.982 0.326304    
    ## BonusMalus114   0.815761   0.707275   1.153 0.248752    
    ## BonusMalus115   2.137912   0.333703   6.407 1.49e-10 ***
    ## BonusMalus116   1.042019   1.000244   1.042 0.297521    
    ## BonusMalus118   1.449159   0.074025  19.577  < 2e-16 ***
    ## BonusMalus119  -9.097942  52.575200  -0.173 0.862615    
    ## BonusMalus120   1.397899   0.577600   2.420 0.015513 *  
    ## BonusMalus125   1.668247   0.070843  23.549  < 2e-16 ***
    ## BonusMalus126   0.431812   1.000120   0.432 0.665916    
    ## BonusMalus132   1.493999   0.258555   5.778 7.55e-09 ***
    ## BonusMalus133   0.995430   0.408545   2.437 0.014829 *  
    ## BonusMalus138  -9.165153 127.039870  -0.072 0.942487    
    ## BonusMalus139   1.383499   0.408615   3.386 0.000710 ***
    ## BonusMalus140   1.805085   0.224061   8.056 7.87e-16 ***
    ## BonusMalus147   1.714452   0.213787   8.019 1.06e-15 ***
    ## BonusMalus148   1.702676   0.353887   4.811 1.50e-06 ***
    ## BonusMalus150   1.740757   0.162766  10.695  < 2e-16 ***
    ## VehBrandB10    -0.043468   0.043473  -1.000 0.317367    
    ## VehBrandB11    -0.003093   0.047585  -0.065 0.948175    
    ## VehBrandB12     0.065756   0.019497   3.373 0.000745 ***
    ## VehBrandB13     0.007426   0.048765   0.152 0.878963    
    ## VehBrandB14    -0.170100   0.090101  -1.888 0.059043 .  
    ## VehBrandB2     -0.008655   0.018219  -0.475 0.634730    
    ## VehBrandB3     -0.015924   0.026165  -0.609 0.542798    
    ## VehBrandB4     -0.019593   0.035561  -0.551 0.581647    
    ## VehBrandB5      0.038506   0.029939   1.286 0.198400    
    ## VehBrandB6     -0.058713   0.034067  -1.723 0.084802 .  
    ## VehGasRegular   0.066168   0.012950   5.109 3.23e-07 ***
    ## Density         0.131611   0.076879   1.712 0.086911 .  
    ## RegionR21       0.075152   0.099299   0.757 0.449156    
    ## RegionR22       0.127401   0.060968   2.090 0.036652 *  
    ## RegionR23      -0.293875   0.070740  -4.154 3.26e-05 ***
    ## RegionR24       0.200263   0.029234   6.850 7.36e-12 ***
    ## RegionR25       0.191405   0.053394   3.585 0.000337 ***
    ## RegionR26       0.010329   0.058230   0.177 0.859211    
    ## RegionR31      -0.210481   0.042525  -4.950 7.44e-07 ***
    ## RegionR41      -0.052665   0.054179  -0.972 0.331017    
    ## RegionR42       0.149337   0.103698   1.440 0.149836    
    ## RegionR43      -0.307140   0.170939  -1.797 0.072370 .  
    ## RegionR52       0.050748   0.036226   1.401 0.161254    
    ## RegionR53       0.302852   0.034348   8.817  < 2e-16 ***
    ## RegionR54       0.036602   0.046165   0.793 0.427867    
    ## RegionR72      -0.140081   0.040062  -3.497 0.000471 ***
    ## RegionR73      -0.184760   0.050763  -3.640 0.000273 ***
    ## RegionR74       0.206630   0.078380   2.636 0.008382 ** 
    ## RegionR82       0.129843   0.029011   4.476 7.62e-06 ***
    ## RegionR83      -0.347772   0.092368  -3.765 0.000167 ***
    ## RegionR91      -0.132232   0.038809  -3.407 0.000656 ***
    ## RegionR93      -0.060171   0.030140  -1.996 0.045895 *  
    ## RegionR94       0.107548   0.078751   1.366 0.172040    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 152137  on 474577  degrees of freedom
    ## Residual deviance: 147821  on 474458  degrees of freedom
    ## AIC: 196570
    ## 
    ## Number of Fisher Scoring iterations: 10

Not the same parameters. Let's compare the deviance of the two models.

``` r
dpois0 <- function(y, mu) {
  d <- rep(1, length(y))
  d[mu != 0] <- dpois(y[mu != 0], lambda = mu[mu != 0])
  
  d
}

dev.loss <- function(y, mu, density.func = dpois0) {
  
  estimated <- mean(log(density.func(y, mu)))
  
  
  best <- mean(log(density.func(y, y)))
  2 * (best - estimated)
}

nn <- predict(model_lr, X_test)
gg <- predict(out, type = "response", newdata = mtpl_test[, c("Area", "VehPower", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "ClaimNb")])

print(dev.loss(mtpl_test$ClaimNb, nn))
```

    ## [1] 0.310165

``` r
print(dev.loss(mtpl_test$ClaimNb, gg))
```

    ## [1] 0.3108266

We get very similar deviance losses, the weights are however not the
same.

We have not taken the cover risk exposure into account but exposure is
usually important in insurance as it gives the time an individual has
been exposed to possible claims. Exposure is usually measured in years.
An individual with 3 claim and 0.5 exposure has a higher claim frequency
than a individual with 3 claims and 1 exposure year.

``` r
Features  <- layer_input(shape = c(ncol(X_train)), dtype = 'float32', name = 'Features') 
Exposure  <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

Network <- Features %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network')

Response <- list(Network, Exposure) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))) ) 
# Note above: second Weight set to zero for bias, if we use the argument use_bias =FALSE then we only need to set array(1, dim = c(1, 1))

model_exposure <- keras_model(inputs = c(Features, Exposure), outputs = c(Response))


model_exposure %>% compile(
  loss = 'poisson',
  optimizer = 'nadam'
)




  fit <- model_exposure %>% fit(
    list(X_train, as.matrix(log(mtpl_train$Exposure))), as.matrix(mtpl_train$ClaimNb),
    epochs = 500,
    batch_size = 10000,
    validation_split = 0.2,
    verbose = 1,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience =5))
  )
```


``` r
out <- glm(ClaimNb ~., data =  mtpl_train[, c("Area", "VehPower", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "ClaimNb", "Exposure")],
           offset = log(Exposure),
           family = quasipoisson())
```

``` r
nn <- predict(model_exposure, list(X_test, as.matrix(log(mtpl_test$Exposure))))
gg <- predict(out, type = "response", newdata = mtpl_test[, c("Area", "VehPower", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region", "ClaimNb", "Exposure")])

print(dev.loss(mtpl_test$ClaimNb, nn))
```

    ## [1] 0.3161008

``` r
print(dev.loss(mtpl_test$ClaimNb, gg))
```

    ## [1] 0.3089037

We get a worse deviance loss.

We have basically performed a glm poisson regression using a NN and
there is nothing “deep” aobut this neural network. To end this study we
add a hidden non-linear layers to try and model non-linearities that may
occur, we then compare it to a Quasi-Poisson glm model.

``` r
Features  <- layer_input(shape = c(ncol(X_train)), dtype = 'float32', name = 'Features') 
Exposure  <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

Network <- Features %>%
    layer_dense(units = 30, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 10, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = 5, activation = 'ReLU', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network')

Response <- list(Network, Exposure) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))) ) 
# Note above: second Weight set to zero for bias, if we use the argument use_bias =FALSE then we only need to set array(1, dim = c(1, 1))

model_deep <- keras_model(inputs = c(Features, Exposure), outputs = c(Response))


model_deep %>% compile(
  loss = 'poisson',
  optimizer = 'nadam'
)




  fit <- model_deep %>% fit(
    list(X_train, as.matrix(log(mtpl_train$Exposure))), as.matrix(mtpl_train$ClaimNb),
    epochs = 500,
    batch_size = 10000,
    validation_split = 0.2,
    verbose = 1,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience =5))
  )
```


``` r
nn <- predict(model_deep, list(X_test, as.matrix(log(mtpl_test$Exposure))))
print(dev.loss(mtpl_test$ClaimNb, nn))
```

    ## [1] 0.3064085

We get slighlty better deviance loss. To find the best model one should
test different number of hidden layers and different activation
functions.

Instead of one-hot encoding our categorical variables, we can use
embeddings. There are multiple possible embeddings that are possible to
use such as pca, t-sne and NLP methods such as word2vec. Keras offers an
Embedding layer that can be used for neural networks on text data, which
is trained during the gradient descent of the NN (not a seperate model).

``` r
X_train <- model.matrix(~-1+ ., mtpl_train[, c("Area", "VehPower", "VehAge" , "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density")])
region_train <- model.matrix(~-1+., mtpl_train[, c("Region"), FALSE])

X_test <- model.matrix(~-1+ ., mtpl_test[, c("Area", "VehPower", "VehAge" ,"DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density")])
region_test <- model.matrix(~-1+., mtpl_test[, c("Region"), FALSE])

Features  <- layer_input(shape = c(ncol(X_train)), dtype = 'float32', name = 'Features') 
Region  <- layer_input(shape = c(1), dtype = 'float32', name = 'Region') 
Exposure  <- layer_input(shape = c(1), dtype = 'float32', name = 'Exposure')

Region_emb <- Region %>% layer_embedding(input_dim = length(unique(mtpl$Region)) , output_dim =2 , input_length =1 , name = "RegionEmbedding") %>% layer_flatten("RegionEmbeddingFlat", data_format = "channels_last")  # data format TensorFlow backend to Keras uses channels last ordering

Network <- list( Features, Region ) %>% layer_concatenate ( name = "NetorkConcatenated") %>%
    layer_dense(units = 30, activation = 'tanh', name = 'layer1') %>%
  layer_dense(units = 10, activation = 'tanh', name = 'layer2') %>%
  layer_dense(units = 5, activation = 'ReLU', name = 'layer3') %>%
  layer_dense(units = 1, activation = 'linear', name = 'Network')

Response <- list(Network, Exposure) %>%
  layer_add(name = 'Add') %>%
  layer_dense(units = 1, activation = "exponential", name = 'Response', trainable = FALSE,
              weights = list(array(1, dim = c(1, 1)), array(0, dim = c(1))) ) 
# Note above: second Weight set to zero for bias, if we use the argument use_bias =FALSE then we only need to set array(1, dim = c(1, 1))

model_deep <- keras_model(inputs = c(Features, Region, Exposure), outputs = c(Response))


model_deep %>% compile(
  loss = 'poisson',
  optimizer = 'nadam'
)



# Note we need to set region to numeric
  fit <- model_deep %>% fit(
    list(X_train, as.matrix(as.numeric(mtpl_train$Region)),  as.matrix(log(mtpl_train$Exposure))), 
    as.matrix(mtpl_train$ClaimNb),
    epochs = 500,
    batch_size = 10000,
    validation_split = 0.2,
    verbose = 1,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience =5))
  )
```

``` r
nn <- predict(model_deep, list(X_test, as.numeric(mtpl_test$Region), as.matrix(log(mtpl_test$Exposure))))
print(dev.loss(mtpl_test$ClaimNb, nn))
```

    ## [1] 0.3124517

Not giving us better performance in this case.
