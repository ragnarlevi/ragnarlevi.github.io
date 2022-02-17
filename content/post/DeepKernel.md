+++
title = "Deep Graph Kernels"
date = "2021-11-16"
author = "Ragnar Levi Gudmundarson"
tags = ["Graph", "Classification", "Kernel"]
+++


# Deep Graph Kernels

In this notebook I will be constructing a deep graph kernel based on [this](https://dl.acm.org/doi/10.1145/2783258.2783417) paper. It utilizes the [Weisfeiler-Lehman](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf) isomorphism test algorithm and/or the [shortest-path ](https://en.wikipedia.org/wiki/Shortest_path_problem) algorithm. The kernel is then used for graph classification. The code is taken from [jcatw](https://github.com/jcatw/scnn/blob/master/scnn/dgk/deep_kernel.py) which again cites Pinar Yanardag as the original author. The code is adjusted to accommodate the networkx library and python 3.


```python
import networkx as nx
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import copy
```

We will use the [mutag](https://paperswithcode.com/dataset/mutag) dataset which is a collection of nitroaromatic compounds and the goal is to predict their mutagenicity on Salmonella typhimurium. First we extract the dataset from the Grakel (a python graph kernel package) and transform it into a list of networkx (python network package) graphs.


```python
from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath

# Loads the MUTAG dataset
MUTAG = fetch_dataset("MUTAG", verbose=False)
Graphs, y_target = MUTAG.data, MUTAG.target

GG = []
for graph in Graphs:
    y = nx.Graph()
    y.add_edges_from(list(graph[0]))
    nx.set_node_attributes(y, graph[1], 'label')
    y =nx.convert_node_labels_to_integers(y)
    GG.append(y)
    
nx.draw_networkx(GG[0], labels = nx.get_node_attributes(GG[0], 'label'))
```


    
![png](DeepKernel_3_0.png)
    


We can create a kernel matrix where all entries belonging to a class are similar to each other and dissimilar to anything else using:

$$K(G,G') = \phi(G)^T M \phi(G')$$

where $M$ is a $|\mathcal{V}| \times |\mathcal{V}|$ psd matrix that encodes the relationship between sub-structures or their proximity where $|\mathcal{V}|$ is the size of the vocabulary. We will let $\phi$ be a the simple the bag of words vector, meaning it counts how often certain sub-structures appear in the graph and is defined as


$$\phi: G \mapsto \phi(G) = (tf(t_1, G), \dots, tf(T_N, G)) \in {R}^N$$

where \\(tf(t_1, G)\\) is the frequency of sub-structure \\(t_1\\) in graph \\(G\\).  If \\(M_{12}\\) is large then sub-structure number 1 and 2 are similar but if it is low then they are dissimilar. Given some sub-structures we want to find a good edit-distance to encode the similarity between two sub-structures. A sub-structure is a very general concept. This means that the user can pre-determine which he or she thinks is important for the classification task at hand. Here I will present sub-structures based on the Weisfeiler-Lehman iteration scheme and the shortest-path.






## WL similarity

The Weisfeiler Lehman iteration is a scheme that test for isomorphisms. If the test is accepted then the graphs **could be** isomorphic and if it is rejected then the graphs are not isomorphic. In words, the algorithm updates the label of the node according to their neighbours. Example, the pairs (a, aabc) and (a, abb) where the first index is the node and the second index are the labels of the neighbours, would receive a different label during this WL iteration but (a, aabc) and (a, aabc) would get the same label. This is much better explained in [this image](https://www.semanticscholar.org/paper/Weisfeiler-Lehman-Graph-Kernels-Shervashidze-Schweitzer/7e1874986cf6433fabf96fff93ef42b60bdc49f8/figure/2).

The algorithm is the following:

![png](capture.png)

The following function builds a corpus using the Weisfeiler Lehman iteration scheme. Multiset labels that belong a given iteration \\(h\\) can be treated as co-occurred in order to partially preserve a notion of similarity. That is for each graph we create a document where the terms in the document are the multi-set labels of the WL-iteration. Thus if two "documents" contain the multi-set labels they should have a high similarity.


```python
def WLSimilarity( X, max_h):
        labels = {}
        label_lookup = {}
        # labels are usually strings so we relabel them as integers for sorting purposes
        label_counter = 0 
        num_graphs = len(X)

        # it stands for wl iteration. the initial labelling is indexed at [-1]
        # contains the feature map
        wl_graph_map = {it: {gidx: dict() for gidx in range(num_graphs)} for it in range(-1, max_h)} # if key error, return 0

      
        # initial labeling
        # label_lookup is a dictionary where key is the label. This loops count how often the each 
        for gidx in range(num_graphs):
            labels[gidx] = np.zeros(len(X[gidx]))
            current_graph_labels = nx.get_node_attributes(X[gidx],'label')
            for node in X[gidx].nodes():
                label = current_graph_labels.get(node, -1) # label of current node, if not labelled it gets -1
                if not label in label_lookup:
                    # if we have not observed this label we relabel at the current label_counter
                    label_lookup[label] = label_counter 
                    labels[gidx][node] = label_counter
                    label_counter += 1
                else:
                    labels[gidx][node] = label_lookup[label]

                # Feature map add 
                wl_graph_map[-1][gidx][label_lookup[label]] = wl_graph_map[-1][gidx].get(label_lookup[label], 0) + 1
        # we are constantly changing the label dictionary so we do a deepcopy
        compressed_labels = copy.deepcopy(labels)


        # WL iterations
        for it in range(max_h):
            label_lookup = {}
            label_counter = 0
            for gidx in range(num_graphs):
                for node in X[gidx].nodes():
                    node_label = tuple([labels[gidx][node]])
                    neighbors = list(X[gidx].neighbors(node))
                    if len(neighbors) > 0:
                        neighbors_label = tuple([labels[gidx][i] for i in neighbors])
                        #node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                        node_label = tuple(tuple(node_label) + tuple(sorted(neighbors_label)))
                    if not node_label in label_lookup:
                        label_lookup[node_label] = str(label_counter)
                        compressed_labels[gidx][node] = str(label_counter)
                        label_counter += 1
                    else:
                        compressed_labels[gidx][node] = label_lookup[node_label]

                    # Add to the feature map 
                    wl_graph_map[it][gidx][label_lookup[node_label]] = wl_graph_map[it][gidx].get(label_lookup[node_label], 0) + 1

            labels = copy.deepcopy(compressed_labels)


        # Create the cropus which contains the documents (graphs)
        # We will also calculate their frequency -> prop_map
        graphs = {}
        prob_map = {}
        corpus = []
        for it in range(-1, max_h):
            for gidx, label_map in wl_graph_map[it].items():
                if gidx not in graphs:
                    graphs[gidx] = []
                    prob_map[gidx] = {}
                for label_, count in label_map.items():
                    label = str(it) + "+" + str(label_)
                    for _ in range(count):
                        graphs[gidx].append(label)
                    prob_map[gidx][label] = count

        prob_map = {gidx: {path: count/float(sum(paths.values())) for path, count in paths.items()} for gidx, paths in prob_map.items()}

        corpus = [graph for graph in graphs.values()]

        return prob_map, corpus, wl_graph_map, graphs
```


```python
prob_map, corpus, wl_graph_map, s = WLSimilarity(GG, 1)
```


```python
corpus[0]
```




    ['-1+0',
       ...
     '-1+0',
     '-1+1',
     '-1+1',
     '-1+2',
     '0+0',
       ...
     '0+0',
     '0+1',
     '0+1',
     '0+1',
     '0+1',
     '0+2',
     '0+3',
     '0+3',
     '0+4']



The first number tells us which WL iteration the multilabel set belongs to, and the second number tells us which multilabel it received. Note that although we see "-1+0" and "0+0", it does not mean that we saw both the multilabel 0 at the initalization and multilabel 0 at the first iteration. The label counter is simply reset to 0 at each WL iteration.

## Shortest path similarity

We can also build shortest path similarity. It is known that the if there is a shortest path between nodes a and b then all sub-paths are also shortest paths. Using this property we can generate "documents" as the collection of all sub-paths in a graph.


```python
def ShortestPathSimilarity(X, cutoff = None):
    """
    Creates shortest path similarity for the Deep Kernel

    :param X: List of nx graphs
    """
    vocabulary = set()
    prob_map = {}
    corpus = []

    for gidx, graph in enumerate(X):

        prob_map[gidx] = {}
        # label of each node
        label_map = list(nx.get_node_attributes(graph,'label').values())
        # get all pairs shortest paths
        all_shortest_paths = nx.all_pairs_shortest_path(graph,cutoff) # nx.floyd_warshall(G)
        # traverse all paths and subpaths
        tmp_corpus = []
        # source is node we are going from
        # sink is node that we are walking to
        for source, sink_map in all_shortest_paths:
            for sink, path in sink_map.items():
                sp_length = len(path)-1
                label = "_".join(map(str, sorted([label_map[source],label_map[sink]]))) + "_" + str(sp_length) 
                tmp_corpus.append(label)
                prob_map[gidx][label] = prob_map[gidx].get(label, 0) + 1
                vocabulary.add(label)
        corpus.append(tmp_corpus)
        # Normalize frequency
    prob_map = {gidx: {path: count/float(sum(paths.values())) for path, count in paths.items()} for gidx, paths in prob_map.items()}

    return prob_map, corpus, label_map
```


```python
prob_map, corpus, label_map = ShortestPathSimilarity(GG)
```


```python
corpus[0]
```




    ['0_0_0',
     '0_0_1',
     '0_0_1',
     '0_0_2',
       ...
     '0_2_6',
     '0_2_7',
     '0_2_7',
     '0_2_8',
     '0_2_8',
     '0_2_9']



We can see how the words in the documents have the form node1_node2_l which means that there exists a path from node1 to node_2 of length l. 

# Building M

We could simply let \\(M = I\\) where \\(I\\) is the identy matrix and than the kernel is simply a bag-of-words kernels, where the vector is a frequency count of the words. We can try to learn the similarity matrix using the [Word2vec](https://arxiv.org/abs/1301.3781) model or more specifically the skip-a-gram version.

The skip-a-gram model hot-encodes our word into the vector $x \in R^N$ where N is the number of words in the vocabulary. Then we define

$$h = W^{(1)}x$$
$$u = W^{(2)}h$$

where \\(W_1 \in R^{N \times L}\\) and \\(W_2 \in R^{L \times N}\\) where \\(M\\) is the dimension of our word representation.

Then the softmax function is applied to \\(y_c = \text{softmax}(u)\\) for \\(c = 1, \dots, C\\) where \\(C\\) is the number of words to predict given word \\(x\\). 

The negative log-likelihood is 

$$\mathcal{L} = -\log \prod_{c=1}^C P(w_{c,i} | w_o) = - \sum_{c=1}^C u_{c, j^*} + \sum_{c=1}^C  \log \sum_{n=1}^N \exp(u_{c,n})$$



\\(j^{*}\\) denotes the ground truth for that given panel and \\(u_{c,n}\\) is the n-th entry in the vector \\(u_c\\).

We want find the weights \\(W^{(1)}\\) and \\(W^{(2)}\\), so we take the derivative with respect to each entry:


\\[\\frac{\\partial \\mathcal{L}}{\\partial W^{(1)}\_{ij}} = \\sum\_{n = 1}^N \\sum\_{c=1}^C \\frac{\\partial \\mathcal{L}}{\\partial u\_{c, n}} \\frac{\\partial u\_{c, n}}{\\partial W^{(1)}\_{ij}}\\]

$$ \frac{\partial \mathcal{L}}{\partial W_{ij}^{(2)}} = \sum_{n = 1}^{N} \sum_{c=1}^{C} \frac{\partial \mathcal{L}}{\partial u_{c, n}} \frac{\partial u_{c, n}}{\partial W^{(2)}_{ij}} $$

We can show that 

$$ \frac{\partial \mathcal{L}}{\partial u_{c, n}} = -\delta_{jj^*}  + y_{c, n} $$

Defining \\(E_n = \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, n}\Big)\\), gives

$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_{ij}^{(1)}} &=  \sum_{n = 1}^N \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, n}\Big) \frac{\partial}{\partial W^{(1)}_{ij}} \Big( \sum_{k=1}^N\sum_{l = 1}^L W_{nk}^{(2)}W_{kl}^{(1)} x_m\Big) \\\\\
&= \sum_{n = 1}^N \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, n}\Big) W^{(2)}_{ni}x_j \\\\\
&= \sum_{n = 1}^N E_n W^{(2)}_{ni}x_j \\\\\
&=  [E^TW^{(2)}]_{i}x_{j}
\end{aligned}$$

and 

$$ 
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_{ij}^{(2)}} &=  \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, i}\Big) \frac{\partial}{\partial W^{(2)}_{ij}} \Big( \sum_{k=1}^N\sum_{l = 1}^L W_{nk}^{(2)}W_{kl}^{(1)} x_l\Big) \\\\\
&=  \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, i}\Big) \sum_{l = 1}^L W_{jl}^{(1)}x_l \\\\\
&=  \sum_{c=1}^C \Big( -\delta_{ii^*}  + y_{c, i}\Big) h_j \\\\\
&=  E_i h_j
\end{aligned}
$$

We the use stochastic gradient descent with learning rate $\eta$ to minimize the loss

$$ W_{ij}^{(1), t} = W_{ij}^{(1), t-1} - \eta  \frac{\partial \mathcal{L}}{\partial W_{ij}^{(1)}} $$
$$ W_{ij}^{(2), t} = W_{ij}^{(2), t-1} - \eta  \frac{\partial \mathcal{L}}{\partial W_{ij}^{(2)}} $$

Finally our similarity matrix is then \\(W_1 W_1^T\\), which can be seen as the the dot product of our word embeddings.


We start by writing a function that hot-encodes our words:


```python
def one_hot_encode(vocab):

    one_hot = dict()
    words = np.array(list(vocab.keys()))

    for k in vocab.keys():
        tmp = np.zeros(len(vocab))
        tmp[k == words] = 1
        one_hot[k] = tmp

    return one_hot
```

Then we write a function that transform our corpus set into a one-hot encoding dataset


```python
# GENERATE TRAINING DATA
def transform_corpus(corpus, one_hot, window = 2):

    training_data = []

    # for each word in each sentence look at window-neighbours and give the pair as training data
    cnt = 1
    for sentence in corpus:
        # print(f'{cnt} {len(corpus)}')
        cnt += 1
        sent_len = len(sentence)

        for i, word in enumerate(sentence):
            
            w_target = one_hot[word]

            w_context = []
            for j in range(i-window, i+window+1):
                if j!=i and j<=sent_len-1 and j>=0:
                    w_context.append(one_hot[sentence[j]])
            training_data.append([w_target, w_context])
    return training_data


def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab

```

Finally we learn the embedding


```python
def forward_pass(W1, W2, x):
    h = np.dot(W1, x)
    u_c = np.dot(W2, h)
    y_c = softmax(u_c)

    # h = np.dot(W1.T, x)
    # u_c = np.dot(W2.T, h)
    # y_c = softmax(u_c)

    return y_c, h, u_c

def softmax(x):
    e_x = np.exp(x - np.max(x))  # - np.max(x) trick for numerical precision
    return e_x / e_x.sum(axis=0)

def backprop( E, h, x, W1 ,W2, eta):

    dw2 = np.outer(E, h)
    dw1 = np.outer(np.dot(E.T, W2), x)

    # UPDATE WEIGHTS
    W1 = W1 - (eta * dw1)
    W2 = W2 - (eta * dw2)


    return W1, W2
```


```python
def train(training_data, L, eta = 0.05, epochs = 20):

    N = len(training_data[0][0])

    # INITIALIZE WEIGHT MATRICES
    W1 = np.random.uniform(-0.7, 0.7, (L, N ))
    W2 = np.random.uniform(-0.7, 0.7, (N, L))


    # CYCLE THROUGH EACH EPOCH
    for i in range(0, epochs):

        loss = 0

        for w_t, w_c in training_data:

            y_pred, h, u = forward_pass(W1, W2, np.array(w_t))

            error = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

            W1, W2 = backprop(error, h, np.array(w_t), W1, W2, eta)

            loss += -np.sum([u[np.where(word == 1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

        print(f'EPOCH: {i}, LOSS: {loss}')


    return W1, W2

```

Let's start with a very simple corpus, too see if you probabilities make sense


```python
corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
vocab = get_vocab(corpus,[])
one_hot = one_hot_encode(vocab)
training_data = transform_corpus(corpus, one_hot, window=2)
```


```python
W1, W2 = train(training_data, 5, eta = 0.01, epochs= 5000)
```

    EPOCH: 0, LOSS: 62.70608789420015
    EPOCH: 1, LOSS: 62.37350155268606
    EPOCH: 2, LOSS: 62.054143476689916
    ...
    EPOCH: 4997, LOSS: 41.05911823870678
    EPOCH: 4998, LOSS: 41.059104040422355
    EPOCH: 4999, LOSS: 41.05908984506421
    


```python
softmax(np.dot(W2, np.dot(W1, one_hot['fox'])))
```




    array([8.68846163e-05, 2.46076725e-01, 2.46746438e-01, 3.80171876e-06,
           2.53355432e-01, 2.52973207e-01, 3.06908315e-04, 4.50604009e-04])




```python
softmax(np.dot(W2, np.dot(W1, one_hot['the'])))
```




    array([1.08782578e-04, 1.60967204e-01, 1.58272813e-01, 9.26069831e-04,
           1.65343960e-01, 1.69151033e-01, 1.74932783e-01, 1.70297355e-01])



## Application of the deep graph kernel in graph classification

Continuing from before we load the data and create our graphs:

First we generate the feature vectors, corpus and vocabulary and generate their one-hot encodings:


```python
prob_map_sp, corpus_sp, label_map_sp = ShortestPathSimilarity(GG, 2)
vocab_sp = get_vocab(corpus_sp,[])
one_hot_sp = one_hot_encode(vocab_sp)
```


```python
len(vocab_sp)
```

    27



Second we create our training data:


```python
training_data_sp = transform_corpus(corpus_sp, one_hot_sp, window = 5)
```


```python
W1_sp, W2_wl = train(training_data_sp, 35, eta = 0.01, epochs= 100)
```

    EPOCH: 0, LOSS: 374814.73546567373
    EPOCH: 1, LOSS: 372198.8859155059
    EPOCH: 2, LOSS: 372248.3620957433
    EPOCH: 3, LOSS: 372257.07566300465
    ...
    EPOCH: 96, LOSS: 371991.4138051814
    EPOCH: 97, LOSS: 371991.1381585368
    EPOCH: 98, LOSS: 371990.7799381037
    EPOCH: 99, LOSS: 371990.3349888323
    

Finally we create our similarity matrix, our feature vectors and the kernel matrix. We will create one with and without the similarity matrix


```python

M = np.dot(W1_sp.T, W1_sp)
P = np.zeros((len(GG), len(vocab_sp)))
for i in range(len(GG)):
    for jdx, j in enumerate(vocab_sp):
        P[i][jdx] = prob_map_sp[i].get(j,0)

K_no_m = P.dot(P.T)
K_m = (P.dot(M)).dot(P.T)
```

With our kernel matrix, we can perform classification:


```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Train an SVM classifier and make predictions
clf = SVC(kernel='precomputed', C= 50)

clf.fit(K_no_m, y_target) 
y_pred_no_m = clf.predict(K_no_m)

clf.fit(K_m, y_target) 
y_pred_m = clf.predict(K_m)

# Evaluate the predictions
print("Accuracy without similarity:", accuracy_score(y_pred_no_m, y_target))
print("Accuracy with similarity:", accuracy_score(y_pred_m,y_target))
```

    Accuracy without similarity: 0.8297872340425532
    Accuracy with similarity: 0.8085106382978723
    

 Let's try classification with the Gensim library


```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=corpus_sp, vector_size=35, window=5, min_count=0, workers=4)

```


```python
M = np.zeros((len(vocab_sp), len(vocab_sp)))
for idx, i in enumerate(vocab_sp):
    for jdx, j in enumerate(vocab_sp):
        M[idx, jdx] = np.dot(model.wv[i], model.wv[j])

K_m2 = (P.dot(M)).dot(P.T)
clf.fit(K_m2, y_target) 
y_pred_m = clf.predict(K_m2)

# Evaluate the predictions
print("Accuracy with similarity Gensim:", accuracy_score(y_pred_m,y_target))
```

    Accuracy with similarity Gensim: 0.7287234042553191
    

The accuracies are very similar and similar to the ones stated in the paper. However the "Deep" in the deep kernel is not performing as well as a simple bag-of-words. Perhaps, the accuracy could be increases with hyper-parameter tunings, such as window-size, word-embedding dimension or number of epochs.

# Improvements

To improve the word to vec model (for more general embeddings) we could do

* Phrase Generation - For example concatenate co-occuring words such as "los" and "angeles" become "los_angeles".
* Subsampling - Decrease the portion of a word that is very frequent to balance between rare and frequent words.
* Negative Sampling - Only consider a small percentage of the words in the error term

During the classification we should do cross-validation to choose hyperparameters

