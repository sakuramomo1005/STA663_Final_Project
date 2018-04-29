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

<img src=https://raw.githubusercontent.com/sakuramomo1005/STA663_Final_Project/master/Data/1.png alt="Drawing" style="width: 600px;"/>

<p style="text-align: center;">
Figure 1: Chinese restaurant process 
</p>
