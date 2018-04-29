# STA663_Final_Project
sta 663 final project with Yiling Liu

### The chosen paper:

Hierarchical Topic Models and the Nested Chinese Restaurant Process


## 1. Introduction
Current statistical modeling tools, in particular, classical model selection techniques which based on hypothesis testing behaves poorly in growing datasets which bring new entities and new structures to the fore. However, recently various domains have dataset with open-ended nature, complex probabilistic models were proposed to address the challenges brought by the growing dataset among which the problem of learning a topic hierarchy from data is very important. For example, if we want to discover common usage patterns or topics in a collection of "documents" which contain a set of words, we need to specify a generative probabilistic model for hierarchical structures and adopt Bayesian perspective to learn such structures from data. One approach is to use Chinese Restaurant Process (CRP) which is a distribution on partitions of integers. CRP treats our hierarchies as random variables and constructs the
hierarchies procedurally based on the available data. It relaxes the constrains of dimensions and uses unbounded dimensionality of classes which means one class from infinite array of classes is assigned to each object. However, in the paper "Hierarchical Topic models and
the Nested Chinese Restaurant Process", we extend the CRP to the Nested Chinese Restaurant Process (nCRP) which is a hierarchy of partitions. Also, it will be applied as a representation of prior and posterior distributions for topic hierarchies. In the nCRP, the constrains of the similarity of distributions associated with parents and children can be relaxed and, in our analysis,, we focuses on the model selection problem.

## 2. Chinese Restaurant Process

The Chinese Restaurant Process (CRP) is a distribuition on partitions of integers. Imagine there are M customers in a Chinese restaurant with infinte tables. The first customer sit in the first table. The following customers have two kinds of choices:

+ Sit in the table that some one alse is already there 
+ Sit in a new table

These two choices have probabilities that depend on the previosu customers at the tables. 
<br>
Specifically, for the $m$th customer, the probability to sit in a table is:
+ p(occupied table i| previous customers) = $\frac{m_i}{\gamma+m-1}$
+ p(next unoccupied table| previous customers) = $\frac{\gamma}{\gamma+m-1}$,

where $m_i$ represnets the number of previous customers at the table $i$; $\gamma$ is a parameter.

If we have M customers, the CRP will give us a partion of M customers, which has the same structure as a Dirichlet process. 

<img src="https://github.com/favicon.ico" width="80">

<img src="https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/1.png" width="200">

<p style="text-align: center;">
Figure 1: Chinese restaurant process 
</p>

Suppose a traveller came to a new city and wanter to try the restaurants there. There is a root restaurant, which is the first stop for new travellers. He came to the root restaurant and chose a table based on the Chinese restaurant process we described before. And each table has a card that references to the next restaurant. The traveller followed the card's instruction and went to the restaurant on the card. Then he chose a table in the second restaurant followed the CRP. In a conclusion, each traveller has a path that contains a batch of restaurants and each restaurant represents a level of the topic model. 

## 3. A hierarchical topic model
In machine learning and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body.Imagine we have a batch of documents, which compose a corpus. The words in the corpus compose the vocabulary. 
To be more concise, we can summarize the topic model and hierarchical topic model as:

### 3.1 A topic model 

Generation of a document:
1. Choose a $K$-vector $\theta$ of topic proportions from a distribution $p(\theta|\alpha)$ 
2. Repeated sample words from the mixture distriubtion $p(\omega|\theta)$ for the chosen value of $\theta$

Besides, when the $p(\theta|\alpha)$ is chosen to be a Dirichlet distribution, these processes are identified as a latent Dirichlet allocation model (LDA). The Figure 3 shows the typical LDA process. 

+ Word: the basic unit from a vocabulary of size $V$ (include $V$ distinct words).
<br>
+ Document: a sequence of N words. $W=[w_1, w_2, . . .,w_N]$
<br>
+ Corpus: a collection of M documents, which is $D=[W_1, W_2,. . . ,W_3]$
<br>
+ $\alpha,\beta$: parameters that specifying the nature of priors on $\theta$ and $\phi$


<img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/3.png alt="Drawing" style="width: 500px;"/>

<p style="text-align: center;">
Figure 3: Plate notation for LDA with Dirichlet-distributed topic-word distribution
</p>

### 3.2 A hierarchical topic model 

Back to the hierarchical topic model, which is very simliar with previous one but added a hierarchical structure. For a hierarchial topic model with L-levels, we can imagine it as a L-level tree and each node presents a topic.

Generation of a document:
1. Choose a path from the root to a leaf
2. Choose the topic proportions $\theta$ from a L-dimension Dirichlet
3. Generated the words in the document for m a mixture of the topics along the path from the root to leaf, wiht mixing proportions $\theta$

This generation of document is very simliar with previous one except the mixing proportion $\theta$ is from a hierarchical structure

<img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/4.png alt="Drawing" style="width: 500px;"/>

<p style="text-align: center;">
Figure 4: hierarchical LDA with a nested CRP prior
</p>

The Figure 4 represents the process of hLDA:

For the $m$th document in the corpus
+ Let $c_1$ be the the root restaurant
+ For each level $l \in \{2,. . ., L\}$: 1. Draw a table from the restaurant $c_{l-1}$. 2. Set $c_l$ to be the restaurant referred to by that table
+ Draw an $L$-dimensional topic proportion vector $\theta$ for $Dir (\alpha)$
+ For each word $n\in \{1, . . ., N\}$: 1. Draw $z \in {1,. . . ,L}$ from $mult (\theta)$; 2. Draw $w_n$ from the topic assoicated with the restaurant $c_z$


## 4. Approximate inference by Gibbs sampling

### 4.1 Introduction to Gibbs sampling

Gibbs sampling is commonly used for statistical inference to determine the
best value of a parameter.  The idea in Gibbs sampling is to generate posterior samples
by sweeping through each variable (or block of variables) to sample from its conditional
distribution with the remaining variables fixed to their current values. For instance, the standard step for Gibbs sampling over a space of variables a, b, c is:
 + Draw a conditioned on b, c
 + Draw b conditioned on a, c
 + Draw c conditioned on a, b
This process continues until “convergence”, which means that the sample values have the same distribution as if they were sampled from the true posterior joint distribution


### 4.2 Gibbs sampling for the hLDA model

**The variables that are needed to be sampled are:**

1. $w_{m,n}$: the $n$th word in the $m$th document (Important note: these are the only observed variables in the model)
2. $c_{m,l}$: the restaurant (node), the $l$th topic in the $m$th document
3. $z_{m,n}$: the assignment of the $n$th word in the $m$th document to one of the $L$ topics
4. There are also some variables needed in the model, but they are not needed to be sampled

After illustrate the variables in the model, we also need to know the order and the methods of the sampling. We can apply the sampling methods into two steps: 
1. sample the $z_{m,n}$ variale by using LDA+CRP
2. sample the $c_{m,l}$ based on the first step (given the LDA hidden variables). 


* To be more specific:

### 4.2.1 Sample $z_{m,n}$

The $z_{m,n}$ is sampled under LDA model based on the method in paper:


\begin{align*}
p(z_{i}=j\hspace{0.5ex}|\hspace{0.5ex}{\bf z}_{-i},{\bf w})\propto\frac{n_{-i,  j}^{(w_{i})}+\beta}{n_{-i, j}^{(\cdot)}+W\beta}\frac{n^{(d_{i})}+\alpha}{n_{-i,\cdot}^{(d_{i})}+T\alpha}
\end{align*}

where:

 + $z_{i}$ is the assignments of words to topics;
 + $n_{-i,j}^{(w_{i})}$ is number of words assigned to topic $j$ that are the same as $w_i$; 
 + $n_{-i,j}^{(\cdot)}$ is total number of words assigned to topic $j$; 
 + $n_{-i,j}^{(d_{i})}$ shows number of words from document $d_i$ assigned to topic $j$, $n_{-i,\cdot}^{(d_{i})}$ represents total number of words in document $d_i$;
 + $W$ shows number of words have been assigned
 + $\alpha,\beta$: free parameters that determine how heavily these empirical distributins are smoothed.
 
 ### 4.2.2 sample $c_m$ from the nCRP

The conitional distibution for $c_m$:

 + $p(w_m|c,w_{-m},z)$: the likelihood of the data given a particular choice of $c_m$
 + $p(c_m|c_{-m})$: the prior on $c_m$ implied by the nested CRP

$$p(c_m | w, c_{-m}, z) \propto p(w_m | c, w_{-m}, z)  p(c_m | c_{-m})$$

The calculation of the $p(w_m | c, w_{-m},z)$ value based on the likelihood function: 

$$p(w_m | c, w_{-m},z) = \prod_{l=1}^{L} (\frac{\Gamma (n_{c_{m,l,-m}}^{(\cdot)}+W\eta)}{\prod_{\omega} \Gamma (n_{c_{m,l,-m}}^{(\omega)}+\eta)}\frac{\prod_{\omega} \Gamma(n_{c_{m,l,-m}}^{(\omega)}+n_{c_{m,l,m}}^{(\cdot)}+\eta)}{\Gamma(n_{c_{m,l,-m}}^{(\cdot)}+ n_{c_{m,l,m}}^{(\cdot)}  W\eta)})$$

where, 

 + $n^{(w)}_{c_m,I,-m}$: the number of word $w$ that have been assigned to the topic indexed y $c_{m,l}$ not including those in the current document
 + $W$: the total vocabulary size
 + $\eta$: free parameter
 
 
## 5. Examples

### 5.1 Hierarchical LDA model results

We fitted our hLDA model to a dataset to demostrate the applicability to real data. Using the data set pubmed.pic from our homework, with a documents of 178 and word vocabulary of 1444. We fisrtly drew a wordcloud plot (Figure 6) to show the frequencies of words. And then we tested the hLDA model and drew the tree plot.A four level hierachy was estimated and shown in Figure 7. Our model captured function words without an auxiliary list which usually required for other models. From the results, we can conclude that hLDA can be a useful approach when dealing with text applications. 

<img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/wordcloud.png alt="Drawing" style="width: 600px;"/>

hLDA results: not show, please see the notebook report

### 5.2 Feasibility for learning text hierarchies in hLDA

To verify the feasibility of nested CRP process for learning text hierarchies in hLDA, we follow the process described in the paper by using a contrived corpus on a small vocabulary. As the paper suggested, we generated a corpus of 100 1000-word documents from a three-level hierarchy with a vocabulary of 25 terms and topics on the vocabulary can be viewed as bars on a 5 × 5 grid in this corpus. The root topic places its
probability mass on the bottom bar. On the second level, one topic is identified with the leftmost bar, while the rightmost bar represents a second topic. From the Figure 5 illustrating six documents sampled from this model, the leftmost topic has two subtopics while the rightmost topic has one subtopic. Also, Figure 6 illustrates the correct hierarchy found by the Gibbs sampler on this corpus.


Figure 5: Six sampled  documents              |  |  | Figure 6: Correct hierarchy
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/5.png alt="Drawing" style="width: 200px;"/>  |   |   | <img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/6.png alt="Drawing" style="width: 200px;"/>


## 6. Profiling and Optimization 
Main functions were written in Python using package numpy. We profiled the codes using cProfile to find the functions or parts of code taking significant amounts of time. The results of the profiler are shown below. We clearly see that most of the computational time was spent on the functions: **nodes**. We should optimize the most time consuming function. Also, to make it faster, we did optimization for all the functions we used in our package. Two approaches were applied to reduce computational times.

### 6.1 Naive Version

For the nested Chinese Restaurant Process, our first naive version codes consisted of 8 main functions. The time is 1.24 s

### 6.2 JIT Optimization
We add the JIT decorator in front of the chunk of functions and the total running time reduced to 885ms.

### 6.3 Cython Optimization
Another way we looked at improving the performance of the code was by cythonizing the code. We rewrote our functions into cython format and the total running time reduced to 856ms.

p.s. The cython codes were deleted since they took too many space. The codes can be find under /Code/Optimization.ipynb

## 7. Unit Testing

Several unit tests were applied to test the validity of the code.
All the tests pass. They are:

1. Test if the likelihood is positive
2. Test the matrixs product with np.dot have the correct size 
3. Test the values in log function are positive
4. Test if each object sampled atleast one feature (which it is supposed to)

## 9. Comparison with Chinese Restaurant Process (CRP)

Chinese Restaurant Process is a clustering algorithm based on objects previously in the cluster. It describes customers' choices of seats from infinite seats in a Chinese Restaurant with infinite tables. Customers come one after another and to start with, first customer sitting on the first table and then each customer chooses an occupied table with probability proportional to the number of customers already sitting on that table and chooses the next vacant table with probability α.The process ends when every customer has seat on a table and it allows for clustering of infinite number of objects into infinite classes.The CRP uses LDA model which needs number of topics as input and the probability of the words in each topic as output. Different from LDA model, our hLDA model applies nonparametric prior which allows sampling the number of topics by nCRP,so we don't need to know the number of topics so it fits the growing data in various fields.Additionally, hLDA model investigates the relationship between topics and words, consider each document without assuming corpus as a big document and returns a topic hierarchy tree instead of a single-layer word distributions for topics.

## 10. Conclusion 

In this project, we presented the nested Chinese restaurant process, a distribution on hierarchical partitions. We presented a Gibbs sampling proecdure for this model to explore the space of the tress and topics. This method has advantages:
+ The nested CRP allows a flexible family of prior distributions over arbitrary
tree structures; definitely could be useful for more than just topic models.
+  Nice qualitative results for topic hierarchies.

Also it has some disadvantages:
+ The restriction that documents can only follow a single path in the tree is a
possibly limiting one.
+ Quantitative evaluation is not extensive enough.

We tested our code by using the example from pubmed and drew the tree plot.However, we can find that in the root level, the words like "of","the","a" we selected. These words may not make sense. Therefore, to improve the model we may need to detect and delete those kinds of words at first. 

## 11.Install the package 

The hierarchical_LDA code of the paper Hierarchical Topic Models and the Nested Chinese Restaurant Process is released on github with the package named hierarchical_LDA. 

One can easily install the package by:

 + pip install --index-url https://test.pypi.org/simple/ hierarchical_LDA

 + Or download from https://github.com/sakuramomo1005/STA663_Final_Project.git and install by running python setup.py install. 

The package provides 4 functions:

   + hierarchical_LDA.CRP_next: 
    + Funcion:  Chinese Restaurant Process
    + Input: 
      + lambdas: concentration parameter 
      + topic: the exist tables 


   + hierarchical_LDA.topics: 
    + Funcion:  sample zmn under LDA model
    + Input: 
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + lambdas: concentration parameter 
      
    
   + hierarchical_LDA.Z: 
    + Funcion:  sample zmn under LDA model
    + Input: 
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + T: the number of topics
      + alpha, beta: parameters 

    
  + hierarchical_LDA.CRP_prior: 
    + Funcion:  Chinese Restaurant Process
    + Input: 
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + topic: the exist topics
      + eta: free parameters

    
   + hierarchical_LDA.nodes: 
    + Funcion:  Gibbs sampling 
    + Input: 
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + T: topic number, artifical
      + iters: iteration times
      + alpha, beta,lambdas, eta: parameters 

    
  + hierarchical_LDA.word_likelihood: 
    + Funcion:  the $p(w_m | c, w_{-m},z)$ likelihood function
    + Input: 
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + topic: the exist topics
      + eta: free parameters
 

   + hierarchical_LDA.hierarchical_LDA: 
    + Funcion:  hierarchical LDA model for topics identify
    + Input: 
      + hlda: the results from the hierarchical_LDA function
      + num: how many words to show in each nodes
      + corpus: the total corpus, a list of documents, that is, a list of lists
      + iters: iteration times
      + level: the tree level
      + num: how many words to show in each nodes
      + alpha, beta,lambdas, eta: parameters 

    
  + hierarchical_LDA.tree_plot: 
    + Funcion for the tree plot generation
    + Input: 
      + hlda: the results from the hierarchical_LDA function
      + num: how many words to show in each nodes

## References
[1] Griffiths, Thomas L., and Mark Steyvers. "A probabilistic approach to semantic representation." Proceedings of the 24th annual conference of the cognitive science society. 2002.

[2] Griffiths, D. M. B. T. L., and M. I. J. J. B. Tenenbaum. "Hierarchical topic models and the nested chinese restaurant process." Advances in neural information processing systems 16 (2004): 17.

[3] Blei, David M., Thomas L. Griffiths, and Michael I. Jordan. "The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies." Journal of the ACM (JACM) 57.2 (2010): 7.

