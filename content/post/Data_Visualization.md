+++
title = "Data Visualization and Data Reduction"
date = "2021-08-30"
author = "Ragnar Levi Gudmundarson"
tags = ["Dimension Reduction", "Visualization", "Data Analysis"]
+++


# Introduction



In this notebook I will be trying out some dimensionality reduction techniques which can be used for modelling and/or data visualization. Dimensional reduction occupies a central position in many fields. In essence, the goal is to change the representation of data sets, originally in a form involving a large number of variables, into a low-dimensional description. The main difference between data reduction and data visualization is that some data visualization techniques can not be used on unseen data, thus they will not be useful in the modelling part. Data visualization is still an import aspect in every modelling task as discovering patterns at an early stage helps to guide the next steps of data science. If categories are well-separated the visualization method, machine learning is likely to be able to find a mapping from an unseen new data point to its value. Given the right prediction algorithm, we can then expect to achieve high accuracy.



The methods can be either linear or nonlinear. Non-linear methods are often used as data points seldom live on linear manifolds. However, non-linear methods can be very sensitive to hyper-parameters. We also want the methods to preserve local structures because in many applications, distances of points that are far apart are meaningless, and therefore need not be preserved. We should keep in mind that since we often have many knobs to tune, we can easily fall into the trap of over-tuning until we see what you wanted to see. This is especially true for the non-linear method.


```python
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA, PCA

from sklearn.manifold import MDS, Isomap

from sklearn.metrics.pairwise import euclidean_distances

from scipy.spatial import distance_matrix

import seaborn as sns

from pandas.api.types import is_numeric_dtype

import matplotlib.patches as mpatches

from scipy.linalg import eigh, svd
```

# Ghoul data

We will work with a very basic data set just to capture the main ideas. In real life however multiple data tweeking and munging needs to be done beforehand, which will probably be more time consuming than the actual modelling part.  We are given data on creatures and the goal is to classify the creature type, they can either be Ghost, Ghoul, or Goblin (https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo/data). We have  4 numerical features: bone_length, rotting_flesh, hair_length, has_soul (percentage of soul in creature) and 1 non-ordinal categorical variable: color (dominant color of the creature):


```python
creatures_train = pd.read_csv("data/creatures_train.csv")

creatures_train.head()

sns.pairplot(creatures_train, hue="type", height=2)



creatures_train_centered = creatures_train.copy()



for col in creatures_train_centered.columns:

    if is_numeric_dtype(creatures_train_centered[col]):

        creatures_train_centered[col] = \

            (creatures_train[col] - creatures_train[col].mean())/np.std(creatures_train[col])



creatures_train_centered.head()



data_numbers = creatures_train_centered[['bone_length', 'rotting_flesh', 

                                         'hair_length', 'has_soul']]



numerical_variables = ['bone_length', 'rotting_flesh', 

                       'hair_length', 'has_soul']

categorical_variables = ['color']



# map each creature to a color

color_map = np.array(['blue'] * creatures_train.shape[0],dtype=object )

color_map[creatures_train['type'] == 'Goblin'] = 'orange'

color_map[creatures_train['type'] == 'Ghost'] = 'green'
```


    
![png](output_5_0.png)
    



```python
def plot_embedding(embedding, color_map, alpha = 1):

    plt.scatter(embedding[:, 0],embedding[:, 1], c=color_map, s=40, cmap='viridis', alpha = alpha)

    pop_a = mpatches.Patch(color='blue', label='Ghoul')

    pop_b = mpatches.Patch(color='orange', label='Goblin')

    pop_c = mpatches.Patch(color='green', label='Ghost')

    plt.legend(handles=[pop_a,pop_b,pop_c])



 
```

In the follwing we let  \\(X\\) be \\(n \\times p\\) matrix where \\(n\\) is the number of data points and \\(p\\) is the number of features.

# PCA



Principal Component Analysis, or PCA, is probably the most widely used embedding. PCA transforms the data into a new coordinate system with an orthogonal linear transformation such that the first coordinate has the greatest variance, the second coordinate has the second greatest variance, and so on. The main advantage is that it is quite simple and fast, but the disadvantage is that it can only capture linear structures, so non-linear information will be lost. PCA can both be used for visualization and modelling.



When performing PCA it is important to normalize the data as we want to minimize the variance. If one feature has higher variance than the other features it will skew the decomposition.

Let \\(h\_w(x)\\) denote a orthogonal projection onto a direction \\(w \\in R^d\\). The empirical variance by the projection is:



\\[ var(h\_w) = \\frac{1}{n} \\sum\_{i} h\_w(x\_i)^2 =  \\frac{1}{n} \\sum\_{i} \\frac{(x\_i^Tw)^2}{||w||^2} = \\frac{1}{n} \\frac{w^T X^T X w}{w^Tw} \\]



where the last term is the Rayleigh quotient. We want to find the i-th principal direction \\(w\_i\\) such that:



\\[ w\_i = \\underset{w \\perp {w\_1, \\dots w\_{i-1}}}{\\operatorname{arg max}} var(h\_w) \\quad s.t. \\quad ||w|| = 1 \\]



The Lagrangian form is given by:



\\[ L(w,\\lambda) = \\frac{1}{n}  w^T X^T X w - \\lambda w^Tw \\]



Taking the derivative yields



\\[ X^T X w\_i =\\lambda w\_i \\]



This is a well studied eigenvalue problem.



Once we have found the directions \\(w\_i\\) we project our data onto the directions \\(w\_i\\) as \\(\\tilde{X} = XW\\), where \\(W\\) is a matrix having \\(w\_i\\) as columns. Each column in \\(\\tilde{X}\\) is called a principal component. The first column being principal component 1 or PC1, and the next column being the principal component 1 or PC2, and so on. The principal components are uncorrelated. \\(Cov(w\_i^T x\_i, w\_j^T x\_j) = w\_i^TCw\_j = \\lambda\_j w\_i^Tw\_j =0\\) (here \\(x\_i\\) and \\(x\_j\\) are a \\(p \\times 1\\) vector, representing a data point. The data has already been normalized in the code above.

The procedure is fairly straightforward to code, we will only take use 2 coordinates but the number of optimal principal direction could be found with a [Scree plot](https://en.wikipedia.org/wiki/Scree_plot) :


```python
C = (1.0/data_numbers.shape[0])*data_numbers.T.dot(data_numbers)



lambda_, alpha = eigh(C)

print(lambda_)



# Do orhogonal projection

coef = np.fliplr(alpha[:,-2:])  # we have to flip the eigenvector matrix as we want the vector corresponding to the highest eigenvalue

X_projected = data_numbers.dot(coef)
```

    [0.51569022 0.63469455 0.97799092 1.87162431]
    

We can check if the projections are orthogonal.


```python
np.corrcoef(X_projected.T)
```




    array([[1.00000000e+00, 7.69730711e-17],
           [7.69730711e-17, 1.00000000e+00]])



We have made them uncorrelated. Very cool. Let's plot. Instead of simple scatter plot, we create a so-called biplot. This plot allows us to see the effect of each variable on each group. 


```python
def myBiplot(score,coeff,labels, color, alpha = 1):

    """

    Function that plots a PCA biplot

    """



    if type(score) != 'numpy.ndarray':

        score = np.array(X_projected)



    xs = score[:,0]

    ys = score[:,1]

    n = coeff.shape[0]



    plt.figure(figsize=(8,5))

    plt.subplot(1, 1,  1)



    

    scalex = 1.0/(xs.max() - xs.min())

    scaley = 1.0/(ys.max() - ys.min())

    points = plt.scatter(xs * scalex, ys * scaley, c = color, alpha = alpha)



    for i in range(n):

        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', alpha = 0.7)

        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'black', ha = 'center', va = 'center')



    pop_a = mpatches.Patch(color='blue', label='Ghoul')

    pop_b = mpatches.Patch(color='orange', label='Goblin')

    pop_c = mpatches.Patch(color='green', label='Ghost')



    plt.legend(handles=[pop_a,pop_b, pop_c])



    plt.xlim([-1,1])

    plt.ylim([-1,1])

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.grid()
```


```python
myBiplot(X_projected, coef, labels = list(data_numbers.columns), color = color_map, alpha = 0.5)
```


    
![png](output_16_0.png)
    


We get a nice visual separation, especially between A pretty Ghouls and Ghosts. We can see that Ghouls are more associated with high values of hair_length, has_soul and bone_length. Also, we see that the first principal component is highly correlated with hair_length, has_soul and bone_length as the the value is greater than 0.5. The second principal component is highly negatively correlated with rotting flesh and bone_length.

 How does our method compare to the scipy method?


```python
pca = PCA(n_components=2, svd_solver='full')

X_projected_pca_scipy = pca.fit_transform(data_numbers)

pca.singular_values_
```




    array([26.350951  , 19.04821856])




```python
myBiplot(X_projected_pca_scipy,np.transpose(pca.components_[0:2, :]), labels = numerical_variables, color = color_map, alpha = 0.5)
```


    
![png](output_20_0.png)
    


This is the same picture, just upside down as the eigenvector signs are irrelevant.

Remark: Scipy uses SVD on X which is another method of getting the PCA score. That is we can perform a SVD on \\(X\\) to obtain \\(X = USV^T\\). From this equation we can obtain the covariance matrix as:



 \\[C = X^TX = \\frac{VSU^TYSV^T}{n-1} = \\frac{VS^2V^T}{n-1}\\]



 so the eigenvalues of C can be obtained as \\(\\lambda\_i = s\_i^2/(n-1)\\) where \\(s\_i\\) is the corresponding eigenvalue of  \\(X\\). The principal components is then obtained as \\(XV = USV^TV = US\\)

# KPCA



Kernel PCA (KPCA) is another dimensionality reduction technique closely related to the PCA. Namely, KPCA uses kernel to map the features in a reproducing kernel Hilbert space which is possibly infinite.  That way we can get a nonlinear dimensionality reduction. The main advantages being that we can capture non-linear data structures. Also, kernels can be defined between abstract objects such as time series, strings, and graphs allowing more general structure. The disadvantage is that it is sensitive top the kernel choice and it's hyper-paramters. Note, that if a linear kernel is used, we simply get the PCA back.  KPCA can both be used for visualization and modelling.





Like in PCA, we work with normalized data.



Now we work with functions \\(\\phi: \\mathcal{X} \\mapsto \\mathcal{X} \\) which maps our data living in the data space \\(\\mathcal{X}\\) to functions living in a repdocuing kernel hilbert space \\(\\mathcal{H}\\) and it is in the space \\(\\mathcal{H}\\) where we find a lower dimensional representation of our data.



For kernel pca we have something very similar to the PCA:



\\[var(h\_f) = \\frac{1}{n} \\sum\_i h\_f(x\_i)^2 =  \\frac{1}{n} \\sum\_i \\frac{<\\phi(x\_i), f>\_H^2}{||f||\_H^2} =  \\frac{1}{n} \\sum\_i \\frac{f(x\_i)^2}{||f||\_H^2} \\]



and we want to solve

\\[f\_i = \\underset{f \\perp {f\_1, \\dots f\_{i-1}}}{\\operatorname{arg max}} var(h\_f) \\quad s.t. \\quad ||f||\_H = 1\\]



Using the representer theorem, we can write the function as \\(f\_i = \\sum\_j \\alpha\_i K(x\_j, \\cdot)\\). The objective becomes



\\[\\alpha\_i = \\underset{\\alpha}{\\operatorname{arg max}} \\alpha^T K^2 \\alpha\\]



such that \\(\\alpha\_i^T K \\alpha\_j, \\quad j = 1, \\dots, i-1\\) (orthogonal) and \\(\\alpha\_i^T K \\alpha\_i\\). We can turn this into an eigenvalue problem by doing a change of variables \\(\\beta = K^{1/2}\\alpha\\) where \\(K^{1/2} = U\\Sigma^{1/2}U^T\\).  The problem now becomes:



\\[\\beta\_i = \\underset{\\beta}{\\operatorname{arg max}} \\beta^T K \\beta\\]



such that \\(\\beta\_i^T K \\beta\_j, \\quad j = 1, \\dots, i-1\\) (orthogonal) and \\(\\beta\_i^T K \\beta\_i\\). The solution is \\(\\beta\_i = u\_i\\) where \\(u\_i\\) is the i-th eigenvalue and \\(\\alpha\_i = U\\Sigma^{-1/2}U^Tu\_i = U^T\\Sigma^{-1/2} [\\dots, 1, \\dots]^T = \\frac{1}{\\sqrt{\\lambda\_i}u\_i}\\)



Note that if we take a linear kernel, that is, \\(f\_w(x) = w^Tx\\) with the norm \\(||f\_w||\_H = ||w||\\) we have:



\\[var(h\_w) = \\frac{1}{n} \\sum\_i \\frac{f(x\_i)^2}{||f||\_H^2} = \\frac{1}{n} \\sum\_i \\frac{(x\_i^Tw)^2}{||w||^2}\\]



so a KPCA with a linear kernel is simply a PCA.



We can choose multiple kernels but for simplicity we can use the radial basis function kernel (rbf kernel). First we write the functions we need. Note we only use the 4 numerical features.



An important remark is that we have obtained orthogonal functions in \\(\\mathcal{H}\\), if we would like to find a representation in the data space \\(\\mathcal{X}\\) we would have to find the inverse of the function \\(f\\). This problem is called the pre-image problem and it is generally hard to solve. Thus, what is instead done in practice to reduce the computational complexity is to simply plot the \\(\\alpha\\) values obtained for the visualization and/or the modelling part. 


```python


def fit( K, nr_eigen, center = True):

    """

    :param nr_eigen: Number of eigenvalues

    :param center: Should kernel matrix be centered?

    """



    if center:

        K = center_kernel_matrix(K)



    alphas_ = obtain_alphas(K,nr_eigen)

    nr_eigen = alphas_.shape[1]



    return K, alphas_





def center_kernel_matrix(K):

    """

    :param K: The kernel matrix we want to center

    :return: centered Kernel matrix

    """

    one_l_prime = np.ones(K.shape) / K.shape[1]

    one_l = np.ones(K.shape) / K.shape[1]

    K = K \

        - np.dot(one_l_prime, K) \

        - np.dot(K, one_l) \

        + one_l_prime.dot(K).dot(one_l)

    return K



def rbf_kernel(X, c):

    """

    :param c: parameter for the RBF Kernel function

    :return: Kernel matrix

    """

    # Compute squared euclidean distances between all samples, 

    # store values in a matrix

    sqdist_X = euclidean_distances(X, X, squared=True)

    K = np.exp(-sqdist_X / c)

    return K





def obtain_alphas(K, n):

    """

    :param n: number of components used 

    :return: returns the first n eigenvectors of the K matrix

    """



    lambda_, alpha = eigh(K, eigvals=(K.shape[0]-n,K.shape[0]-1))



    alpha_n = alpha / np.sqrt(lambda_)



    # Order eigenvalues and eigenvectors in descending order

    lambda_ = np.flipud(lambda_)

    alpha_n = np.fliplr(alpha_n)







    return alpha_n 
```


```python
plt.figure(figsize=(20, 5))



for i, rbf_constant in enumerate([0.5, 1, 2, 4]):

    plt.subplot(1, 4, i + 1)

    

    K = rbf_kernel(data_numbers, rbf_constant)

    K, alphas = fit(K, 2)



    plot_embedding(alphas, color_map)

    plt.title(f'rbf constant = {rbf_constant}')
```


    
![png](output_25_0.png)
    


Again we get some distiction between the types. Let's try the Sklearn KPCA


```python
transformer = KernelPCA(n_components = 2, kernel='rbf', fit_inverse_transform=True, gamma = 1/2)

X_kpca = transformer.fit_transform(data_numbers)

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

plot_embedding(transformer.alphas_, color_map)


```


    
![png](output_27_0.png)
    


This is exactly the same figure. We could in theory include categorical variables by using a kernel for categorical data, for example the Aitchison and Aitken kernel function. That is, we can define a kernel between the numerical features with \\(k\_{num}\\) and we also define a different kernel between the categorical features \\(k\_{cat}\\). Then we perform KPCA on \\(k = k\_{num}k\_{cat}\\).



# Multidimensional Scaling



## The classical mds



[Multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling) (MDS) is a data visualization technique which tries to preserve distances between samples globally using only a table of distances between them. That is we are given distances \\(d\_{ij}\\) and we want to find embeddings \\(y\_i \\in R^q\\) for each data point s.t.



\\[ \\sum\_{i <j} (||y\_i - y\_j|| - d\_{ij})\\]



is minimized. We can derive the algorithm, Assume that \\(x \\in R^q\\), \\(p<q\\) :



We know that \\(d\_{ij}^2 =||x\_i - x\_j||^2 = x\_i^Tx\_i + x\_j^Tx\_j - 2x\_i^Tx\_j\\). Let \\(K = XX^T\\) be the Gram matrix (linear kernel). Rewriting the previous equations as \\(d\_{ij} = k\_{ii} + k\_{jj} -2k\_{ij}\\) and assuming that that the data is centered, \\(sum\_{i = 1}^n x\_{im} = 0\\) (any center location could be used as if x is solution then x +c is solution is well) then  for all \\(m\\), we obtain:



\\[\\sum\_{i = 1}^n k\_{ij} = \\sum\_{i = 1}^n \\sum\_{m = 1}^q x\_{im} x\_{jm} = \\sum\_{m = 1}^q x\_{jm} \\sum\_{i = 1}^n x\_{im} = 0\\]



We also have



\\[ \\sum\_{i = 1}^n d\_{ij}^2 = trace(K) + nb\_{jj}, \\quad \\sum\_{j = 1}^n d\_{ij}^2 = trace(K) + nb\_{ii}, \\quad \\sum\_{i = 1}^n \\sum\_{j = 1}^n d\_{ij}^2 = 2n*trace(B) \\]



Rearranging the previous equation and plugging it into \\(d\_{ij} = k\_{ii} + k\_{jj} -2k\_{ij}\\) gives



\\[ k\_{ij} = -1/2 (d\_{ij}^2 - 1/n \\sum\_{i = 1}^n d\_{ij}^2 - 1/n \\sum\_{j = 1}^n d\_{ij}^2  + 1/n^2 \\sum\_{i = 1}^n \\sum\_{j = 1}^n d\_{ij}^2 )\\]





which can be written in a matrix form (try by writing it down for a 3x3 matrix):



\\[K = -1/2 H D^{(2)} H\\]



where \\(H = I - 1/n ee^T\\) and \\(e\\) is a vector of one's. The solution is then given by the eigen-decomposition of \\(K\\), \\(K = V\\Lambda V^T\\) and \\(X = V \\Lambda^{1/2}\\). For dimensionality reduction, simply pick the top \\(q\\) eigenvector and eigenvalues and calculate the embedding as \\(\\tilde{X} = V\_q\\Lambda\_pq{1/2}\\). The disadvantage is that is does not support out-of-sample transformation (we would have to redo the embedding scheme for each new sample). Note, that we can use which ever distance matrix we like so we are not restricted by the euclidean distance.



## The metric mds



Is a generalization of the classical mds which allows for a variety of loss functions. The most widely used loss function is called s tress. see the wiki for more information.



## The Isomap



Isomap generalizes the metric mds where the idea is to perform MDS not on the input space but rather on the geodesic space of the nonlinear data manifold. The first step is to find the nearest neighbours of each data point in high-dimensional data space, this generates a graph where each node is a data point and the edges connect the nearest neighbours of each data point and the weight of the edge is the input space distance.  The second step is to calculate the geodesic pairwise distances between all points by using the a shortest path algorithm, for example, Dijkstra�s algorithm or Floyd�s algorithm. Finally the metric mds embedding is used. The disadvantage is of isomats are potential �short-circuits�, in which a single noisy datapoint provides a bridge between two regions of dataspace that should be far apart in the low-dimensional representation.



The disadvantage of these methods is that they are generally not used on unseen data although it [can be done](https://papers.nips.cc/paper/2003/file/cf05968255451bdefe3c5bc64d550517-Paper.pdf)


```python
def mds(D, p):

    """

    Calculate multidimensional scaling

    :param D: The distance matrix where each entry is squared

    :param p: Embedding dimension

    """

    # We just need to calculate



    n = D.shape[0]

    H = np.identity(n) - (1.0/n) * np.ones((n,n))



    K = -0.5 * np.dot(H,D).dot(H)



    lamda, V = eigh(K)



    # Order eigenvalues and eigenvectors in descending order

    lamda = np.flipud(lambda_)

    V = np.fliplr(V)



    embedding = np.dot(V[:, :p], np.diag(lamda[:p]))



    return embedding
```


```python
# using the  Minkowski p-norm 

plt.figure(figsize=(20, 5))



for i, p in enumerate([1, 2, 3]):

    plt.subplot(1, 3, i+1)

   

    D = distance_matrix(data_numbers, data_numbers, p=p)

    D_sq = np.power(D, 2)



    embedding = mds(D_sq, 2)

    

    plot_embedding(embedding, color_map)

    plt.title(f'Minkowski {p}-norm')
```


    
![png](output_32_0.png)
    


Note that that \\(p=2\\) results in the same embedding as the PCA. This happens as the \\(p =2\\) norm is the same as a linear kernel

Comparing this to Sklearn's mds.


```python
embedding = MDS(n_components=2)

X_transformed = embedding.fit_transform(data_numbers)
```


```python
plot_embedding(X_transformed, color_map)
```


    
![png](output_36_0.png)
    


This is not the same embedding as the sklearn uses the metric mds which uses the SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm to minimizes an objective function.



We can also look at the Isomap embeddings using sklearn.


```python
embedding_isomap = Isomap(n_components=2, n_neighbors=40)

X_transformed_isomap = embedding_isomap.fit_transform(data_numbers)
```


```python
plot_embedding(X_transformed_isomap, color_map)
```


    
![png](output_39_0.png)
    


# Correspondence Analysis





[Correspondence Analysis](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/connection-between-correlation-and-contingency/D3A75249B56AF5DDC436938F1B6EABD1) is a method that helps us finding a connection between correlation and contingency. These methods are best suited for categorical data, but numerical data could be made categorical by binning strategies. The wiki page does a good job explaining the methodology [CA](https://en.wikipedia.org/wiki/Correspondence_analysis).



We start by calculating the contingency table \\(C\\).



We compute the row and column weights, \\(w\_m = \\frac{1}{n\_c} C 1\\) and \\(w\_n = \\frac{1}{n\_c} 1^T C\\). Let \\(n\_c = \\sum \_j \\sum \_j C\_{ij}\\).



The weights are transformed into matrices, \\(W\_m = diag(1/\\sqrt(w\_m))\\) and \\(W\_n = diag(1/\\sqrt(w\_n))\\). Next, the standardized residuals are found



\\[ S = W\_M(P-w\_m w\_n)W\_n\\]



To calculate the coordinates for CA we perform a SVD decomposition of the standardized residual. The left singular vectors correspond to the categories in the rows of the table, and the right singular vectors correspond to the columns. The eigenvalue gives us the variance of each dimension. That is the principal coordinates of the rows are \\(F\_m = W\_m U \\Sigma\\) and the  principal coordinates for the columns are \\(F\_n = W\_n V \\Sigma\\)



We can also plot the data in standard coordinates which are defined as \\(G\_m = W\_mU\\), \\(G\_n = W\_n V\\)



Note when we plot the data in the same space using the principal coordinates (\\(F\\)), we can only interpret the distance between row points or the distance between column points (intra distance). We can not interpret the distance between rows and column points (inter distance). To interpret the distance between column points and row points, we plot the decomposition in an asymmetric fashion. Rows (or columns) points are plotted from the standard coordinates and the profiles of the columns (or the rows) are plotted from the principal coordinates (\\(F\\) and \\(G\\)). If we plot using the standard coordinates (G) then the intra distance will be exaggerated.






```python
def CA(data,row, col):

    """

    :param row: name of row variable

    :param col: name of col variable

    """

    ""

    C = pd.crosstab(data[row],data[col], margins = False)

    print(C)



    col_var = list(C.columns)

    row_var = list(C.index)



    n_c = np.sum(np.sum(C))  # sum of all cell values in C



    w_m = np.array(C).dot(np.ones((C.shape[1], 1))) / n_c  # column weights



    w_n = np.ones((1, C.shape[0])).dot(np.array(C)) / n_c



    W_m = np.diag(1.0/np.sqrt(w_m[:,0]))

    W_n = np.diag(1.0/np.sqrt(w_n[0,:]))



    P = (1/n_c) * C



    # standarized residuals

    S = W_m.dot(P - np.dot(w_m, w_n)).dot(W_n)





    U, Sigma, Vt = svd(S, full_matrices=False)



    F_m = W_m.dot(U).dot(np.diag(Sigma))

    F_n = W_n.dot(Vt.T).dot(np.diag(Sigma))



    G_m = W_m.dot(U)

    G_n = W_n.dot(Vt.T)



    plt.scatter(G_m[:,0], G_m[:,1], c = 'blue')

    for i, name in enumerate(row_var):

        plt.text(G_m[i,0], G_m[i,1], name)



    # plit color 

    plt.scatter(F_n[:,0], F_n[:,1], c = 'red')

    for i, name in enumerate(col_var):

        plt.text(F_n[i,0], F_n[i,1], name)





    return F_m, F_n, G_m, G_n
```

We plot the first two dimensions of each type and each color.


```python
F_m, F_n, G_m, G_n = CA(creatures_train,'type', 'color')


```

    color   black  blood  blue  clear  green  white
    type                                           
    Ghost      14      6     6     32     15     44
    Ghoul      14      4     6     42     13     50
    Goblin     13      2     7     46     14     43
    


    
![png](output_43_1.png)
    


This is pretty anti-climatic, but we could say that blood corresponds mostly with Ghost. I don't these results are so surprising as the the color distribution for the types are very similar if we look at the contingency table. If we would have more than one categorical variable we could use [Multiple correspondence analysis](https://en.wikipedia.org/wiki/Multiple_correspondence_analysis), which is pretty similar to CA and straightforward to implement.

# Diffusion map

[Diffusion map](https://www.sciencedirect.com/science/article/pii/S1063520306000546) (ses also [An Introduction to Diffusion Maps](https://inside.mines.edu/~whereman/talks/delaPorte-Herbst-Hereman-vanderWalt-DiffusionMaps-PRASA2008.pdf)) is data visualization reduction technique. Similar to the Isomap it is a technique relying on a graph based algorithm, interpreting weighted graphs as a notion of geometry.



The diffusion map is generally not used on unseen data but it [can be done](https://papers.nips.cc/paper/2003/file/cf05968255451bdefe3c5bc64d550517-Paper.pdf)





The connectivity between two data points, \\(x\\) and \\(y\\) is defined as the probability of jumping from \\(x\\) to \\(y\\) in one step of a random walk on a graph. The connectivity is expressed in terms of a kernel \\(k(x,y)\\)



The Gaussian kernel (rbf kernel)



\\[k(x,y) = \\exp(-\\frac{||x-y||^2}{\\alpha})\\]



is popular as it gives a prior to the local geometry of \\(X\\) using the euclidean norm. By tweaking the parameter \\(\\alpha\\) we are essentially defining the size of the neighbourhood area.



Then \\(p(x,y)\\) is defined as



\\[p(x,y) = \\frac{1}{d\_X}k(x,y)\\]



where \\(d\_X = \\sum\_{y }k(x,y)\\) is a normalizing constant. Note that \\(p(x,y)\\) is no longer symmetric but it defines a Markov chain, and a path between two points is now measured by the path length, that is, how probable is it to transit from \\(x\\) to \\(y\\) in \\(t\\) steps. Using this idea  we define the diffusion distance of two points in the space as the probability of jumping between them in \\(t steps\\)



\\[ D\_t(X\_i, X\_j) = \\sum\_u | p\_t(X\_i, u) - p\_t(X\_j, u)| ^2\\]



where \\(t\\) denotes the number of steps. As the walk goes on it reveals the geometric structure of the data and the main contributors to the diffusion distance are paths along that structure.



Finally, to visualize the data manifold, we want to find points \\(Y\_i\\) and \\(Y\_j\\), that exist in an \\(r\\)-dimensional euclidean space which conserve the distance between \\(X\_i\\) and \\(X\_j\\) on the manifold, that is.



\\[ || Y\_i - Y\_j ||\_2 = D\_t(X\_i, X\_j) \\]



To find the data points \\(Y\_i\\), we simply pick the top \\(r\\) eigenvalues of \\(P\\) and set \\(Y\_i = [\\lambda\_1 v\_1[i], \\dots, \\lambda\_r v\_1[r]]\\) where \\(v\_n[i]\\) is \\(i\\)-th component of the \\(n\\)-th right eigenvector. A remark is that we look at \\(P' = D^{-1/2} k D^{-1/2}\\), which has the same eigenvalues as \\(P\\), see [An Introduction to Diffusion Maps](https://inside.mines.edu/~whereman/talks/delaPorte-Herbst-Hereman-vanderWalt-DiffusionMaps-PRASA2008.pdf) for a better explanation.



The code implementation is pretty straight forward, it yet again bowls down to an eigenvalue decomposition,


```python
def DiffusionMap(X, alpha, t):

    """



    :param X: data matrix

    :param alpha: Gaussian kernel parameter

    """





    sqdist_X = euclidean_distances(X, X, squared=True)

    K = np.exp(-sqdist_X / alpha)







    d_x = np.sum(K,axis= 0)

    D_inv = np.diag(1/d_x)



    P = np.dot(D_inv,K)



    D_sq_inv = np.diag((d_x) ** -0.5)





    P_prime = np.dot(D_sq_inv, np.dot(K,D_sq_inv))

    P_prime = np.matmul(np.diag((d_x) ** 0.5), np.matmul(P,D_sq_inv))



    lamda, U = eigh(P_prime)



    idx = lamda.argsort()[::-1]

    lamda = lamda[idx]

    U = U[:,idx]

    # print(lamda)



    coord = np.dot(D_sq_inv, U)#.dot(np.diag(lamda ** t))



    return coord, lamda, U
```


```python
embedding_diffusion, lamda, U = DiffusionMap(data_numbers, 1.5, 1)

embedding_diffusion.shape
```




    (371, 371)




```python
plot_embedding(embedding_diffusion[:, 1:], color_map, alpha = 0.7)
```


    
![png](output_50_0.png)
    


# TSNE

[t-sne](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) is another data visualization technique. It maps the data points in such a way that similar points are grouped together with a high probability, and that distant points will be distant with high probability. Therefore, t-sne aims to perserve local structure. The main idea is that the similarity between a point \\(x\_i\\) and all other points \\(x\_j\\) is measured with a conditional Gaussian, \\(p\_{j|i}\\). That is, the probability that \\(x\_i\\) would pick \\(x\_j\\) as a neighbour,



\\[p\_{j|i} \\frac{\\exp (-||x\_i - x\_j||^2/2\\sigma\_i^2)}{\\sum\_{k \\neq i}\\exp (-||x\_i - x\_k||^2/2\\sigma\_i^2}\\]



we set \\(p\_{i|i}\\) to zero. We will explain the \\(\\sigma\\) parameter later on. Then, to introduce symmetry, we will work the probability



\\[ p\_{ij} = \\frac{p\_{j|i} + p\_{i|j}}{2n} \\]



where \\(n\\) is the number of data points.  The denominatior is introcuded to ensure that each data point contributes to the cost function, \\(\\sum\_j p\_{ij} > 1/(2n)\\).



The similarity in the low dimensional map is defined using a Cauchy distribution.



\\[q\_{ij} = \\frac{(1+ ||y\_i - y\_j||)^{-1}}{\\sum\_{k \\neq l} (1+ ||y\_k - y\_l||)^{-1}}\\]



To measure the faithfulness with which \\(q\_{ij}\\) models \\(p\_{ij}\\), the Kullback-Leibler divergence is used.



\\[C = \\sum\_i KL(P\_i || Q\_i) = \\sum\_i \\sum\_j p\_{ji} \\log \\frac{p\_{ji}}{q\_{ji}}\\]



We are interested in finding the best \\(y\_i\\) to represent \\(x\_i\\) in low-dimension. Therefore, we take the gradient with respect to \\(y\_i\\) and we get (see the derivation in the paper)



\\[ \\frac{\\partial C}{\\partial y\_i} = 4 \\sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)(1 + ||y\_i - y\_j||^2)^{-1}\\]



and finally we perform we use gradient descent, with a momentum term to speed up the optimization and to avoid poor local minima.



\\[ y^{(t)} = y^{(t-1)} + \\eta \\frac{\\partial C}{\\partial y} + \\alpha(t)(y^{(t-1)} - y^{(t-2)}) \\]



where \\(\\eta\\) is the learning rate and \\(\\alpha\\) is the momentum at iteration \\(t\\).



The low dimensional embedding is initialized with a normal distribution.



It is important to note that the cost function is non convex and the solution depends on the initial embedding. Therefore, each time the algorithm is run, the outcome won't be the same.



A major weakness of t-SNE is that the cost function is not convex, as a result of which several optimization parameters need to be chosen (including random initial low dimensional embeddings). Therefore, each time the algorithm is run, the outcome won't be the same. That being said, a local optimum of a cost function that accurately captures a good representation is often preferable to the global optimum of a cost function that fails to capture important aspects.





What about \\(\\sigma\_i\\)? It is not likely that there is a single value of \\(\\sigma\_i\\) that is optimal for all datapoints in the data set because the density of the data is likely to vary In dense regions, a smaller value of \\(\\sigma\_i\\) is usually more appropriate than in sparser regions. \\(\\sigma\_i\\) is found by introducing a parameter called perplexity:



\\[ perp(p\_i) = 2^{H(p\_i)}\\]



where 



\\[ H(p\_i) = - \\sum\_{j} p\_{j|i} \\log\_2 p\_{j|i} \\]



We can find \\(\\sigma\_i\\) with a binary search.



The perplexity can be interpreted as a smooth measure of the effective number of neighbors. The performance of t-SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50.

For large data sets, it will be costly to store the probability if each data point considered all other data points as a neighbour. To make the computations feasible we can start by choosing a desired number of neighbors and creating a neighborhood graph for all of the datapoints. Although this is computationally intensive, it is only done once. This is very similar to the isomap and the diffusion map. Finally, \\(p\_{j|i}\\) is defined as the fraction of random walks starting at landmark point \\(x\_i\\) that terminate at landmark point \\(x\_j\\). 


```python
def sigma_binary_search(perplexity, sqdist, n_steps = 100, tolerance = 1e-5) -> np.array:

    """

    Perform binary search to find the sigmas. We perform the search on log scale. Note, we return the probability matrix P



    :param perplexity: perplexity

    :param sqdist: array (n_samples, n_neighbours), Square distances of data points

    :param n_steps: Number of binary search steps.n_steps

    """



    n_samples = sqdist.shape[0]

    n_neighbours = sqdist.shape[1]



    desired_entropy = np.log(perplexity)



    P = np.zeros((n_samples, n_neighbours), dtype=np.float64)



    # Perform binary search for each data point

    for i in range(n_samples):

        # var = 1/(2sigma^2)

        var_min = -np.Inf

        var_max = np.Inf

        var = 1.0



        # start binary search

        for k in range(n_steps):

            

            sum_p_i = 0.0  # the sum of $ p_{j|i}



            for j in range(n_neighbours):

                if j != i:

                    P[i, j] = np.exp(-sqdist[i, j] * var)

                    sum_p_i += P[i, j]



            

            sum_disti_Pi = 0.0

            for j in range(n_neighbours):

                P[i, j] /= sum_p_i

                sum_disti_Pi += sqdist[i, j] * P[i, j]





            entropy = np.log(sum_p_i)*1.0 + var * sum_disti_Pi

            entropy_diff = entropy - desired_entropy



            if np.abs(entropy_diff) <= tolerance:

                break



            # perform binary search

            if entropy_diff > 0.0:

                var_min = var

                if var_max == np.Inf:

                    var *= 2.0

                else:

                    var = (var + var_max) / 2.0

            else:

                var_max = var

                if var_min == -np.Inf:

                    var /= 2.0

                else:

                    var = (var + var_min) / 2.0



    return P
```


```python
def tsnse(X, perplexity, nr_steps, eta = 100, n_steps = 100, tolerance = 1e-5):

    """



    Some of the code is from the author of t-sne https://lvdmaaten.github.io/tsne/

    

    :param X: data matrix (n_samples, n_features)

    :param perplexity: perplexity

    :param n_steps: Number of graident descent steps

    :param eta: Learning rate

    :param sqdist: array (n_samples, n_neighbours), Square distances of data points

    :param n_steps: Number of binary search steps.n_steps

    """



    n = X.shape[0]



    sqdist_X = euclidean_distances(X, X, squared=True)

    P = sigma_binary_search(perplexity, sqdist_X)



    P = (P + P.T)/(2.0 * n)

    P = np.maximum(P, 1e-12)



    # initalize Y

    Y = np.random.randn(n, 2)

    gradient = np.zeros((n, 2))  

    iY_1 = np.zeros((n, 2))





    # start gradient descent

    for t in range(nr_steps):



        Q = np.zeros((n,n))



        sqdist_Y = euclidean_distances(Y, Y, squared=True)

        for i in range(n):

            for j in range(n):

                Q[i,j] = 1./(1. + sqdist_Y[i,j])



        Q[range(n), range(n)] = 0  # set diagnonal as 0

        num = Q.copy()

        Q = Q / np.sum(Q)

        Q = np.maximum(Q, 1e-12)





        # Compute gradient

        PQ = P - Q

        for i in range(n):

            tmp = np.tile(PQ[:, i] * num[:, i], (2, 1))  #(p_ij - q_ij)(1 + ||y_i - y_j||)^-1 part

            gradient[i, :] = np.sum(tmp.T * (Y[i, :] - Y), 0)



        # Perform the update

        if t < 250:

            momentum = 0.5

        else:

            momentum = 0.8



        iY_2 = iY_1.copy()

        iY_1 = Y.copy()

        Y = Y - eta*gradient + momentum * (iY_1 - iY_2)





        # if (t + 1) % 10 == 0:

        #     C = np.sum(P * np.log(P / Q))

        #     print("Iteration %d: error is %f" % (t + 1, C))



    return Y
```


```python
Y = tsnse(data_numbers, 30, 1000, eta = 200)
```


```python
plot_embedding(Y, color_map)
```


    
![png](output_58_0.png)
    


The code seems to be working but it is quite slow. If one wants to do t-sne for data visualization, there is a module in the sklearn library, which is significantly faster.


```python
from sklearn.manifold import TSNE

embedding_tsne = TSNE(n_components=2, learning_rate=200, perplexity=30, early_exaggeration=12).fit_transform(data_numbers)
```


```python
plot_embedding(embedding_tsne, color_map)
```


    
![png](output_61_0.png)
    

