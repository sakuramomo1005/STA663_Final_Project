
import numpy as np
from scipy.special import gammaln
import random
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import graphviz
import pydot
import pygraphviz

def CRP_next(lambdas,topic):
    """
    Description
    ---------
    Funcion: Chinese Restaurant Process
    
    Parameter
    ---------
    alpha: concentration parameter 
    topic: the exist tables 
    
    Return
    ------
    p: the probability for a new customers to sit in each of the tables
    """
    import numpy as np
    N=len(topic) # number of tables
    word_list=[] # total customers
    for t in topic:
        word_list=word_list+t
    m=len(word_list) # customers' number
    
    tables = np.array([len(x) for x in topic])  # tables with their customers
    p_old=tables/(lambdas+m) # the probability of sitting in a table with other people   
    p_new=lambdas/(lambdas+m)      # the probability of sitting in a new table
    p=[p_new]+list(p_old)  # the last probability is the probability to sit in a new table 
    return(p)


def topics(corpus,lambdas):
    topic=[]
    for docs in corpus:
        for word in docs:
            p=CRP_next(lambdas,topic)
            position=np.random.multinomial(1, list((np.array(p)/sum(p))))
            position=int(np.where(position!=0)[0])
            if position==0:
                topic.append([word])
            else:
                topic[position-1].append(word)
    return(topic)


def Z(corpus, T, alpha, beta):
    """
    Description
    ---------
    Funcion:  sample zmn under LDA model
    
    Parameter
    ---------
    corpus: the total corpus, a list of documents, that is, a list of lists
    T: the number of topics
    alpha, beta: parameters
    
    Return
    ------
    topic: the word list in each topic
    topic_num: the length of each topic
    """
    import numpy as np
    W=np.sum([len(word) for word in corpus]) # the number of the total words
    N=len(corpus)                            # the number of documents 
    topic=[[] for t in range(T)]
    topic_num=[[] for t in range(T)]
    for i,di in enumerate(corpus):
        for wi in di:
            p=np.zeros(T)
            for j in range(T):
                nij_wi=topic[j].count(wi)   # number of wi tht assigned to topic j
                nij=len(topic[j])           # total number of words assigned to topic j 
                nij_di=np.sum(np.isin(topic[j],di)) # number of words from di in topic j
                ni_di=len(di)               # total number of words in di
                part1=(nij_wi+beta)/(nij+W*beta)
                part2=(nij_di+alpha)/(ni_di+T*alpha)
                p[j]=part1 * part2
            pp=p/np.sum(p)
            w_assign=np.random.multinomial(1, pp, size=1)
            i_topic=int(np.where(w_assign[0]==1)[0])
            topic[i_topic].append(wi)
            topic_num[i_topic].append(i)
    return(topic,topic_num)

def CRP_prior(corpus,topic,lambdas):
    res=np.zeros((len(corpus),len(topic)))
    for i,docs in enumerate(corpus):
        p_topic=[]
        for j in range(len(topic)):
            temp=[]
            for x in topic[j]:
                if x != i:
                    temp.append(x)
            p_topic.append(temp)
        temp=CRP_next(lambdas,p_topic)
        res[i,:]=temp[1:]   
    return(res)


def word_likelihood(corpus,topic,eta):

    import math
    import numpy as np
    from scipy.special import gammaln

    res=np.zeros((len(corpus),len(topic)))  # generate the results matrix
    
    word_list=[]                            # generate the word list that contains all the words
    for i in range(len(corpus)):
        word_list=word_list+corpus[i]
    W=len(word_list)                        # the length of word list
    
    for i,di in enumerate(corpus):
        p_w=0
        for j in range(len(topic)):         #calculate the tow parts of the equation
            nc_dot=len(topic[j])    
            part1_denominator=1
            part2_nominator=1
            
            overlap=len(set(topic[j]))-len(set(topic[j])-set(di))
            
            part1_nominator = gammaln(nc_dot-overlap+W*eta)
            part2_denominator = gammaln(nc_dot+W*eta)
        
            for word in di:
                ncm_w=topic[j].count(word)-di.count(word)
                if ncm_w <0:
                    ncm_w=0
                nc_w=topic[j].count(word)
                part1_denominator=part1_denominator+gammaln(ncm_w+eta)
                part2_nominator=part2_nominator+gammaln(nc_w+eta)
           
            p_w=part1_nominator-part1_denominator+part2_nominator-part2_denominator 
            res[i,j]=p_w
        res[i, :] = res[i, :] + abs(min(res[i, :]) + 0.1)
    res=res/np.sum(res,axis=1).reshape(-1,1)
    return(res)


def nodes(corpus,T,alpha,beta,lambdas,eta,iters=100):

    word_list=[]
    for x in corpus:
        word_list=word_list+x
    W=len(word_list)
    gibbs=np.zeros((W,iters))
    
    for j in range(iters):
     #   print('iters % j complete', j)
        topic=Z(corpus, T, alpha, beta)[0]
        w_m=word_likelihood(corpus,topic,eta)
        c_=CRP_prior(corpus,topic,lambdas)
        c_m = (w_m * c_) / (w_m * c_).sum(axis = 1).reshape(-1,1)
        
        g=[]
        for i,docs in enumerate(corpus):
            if np.sum(c_m[i,:-1])>1:
                c_m[i,-1]=0
                c_m[i,:-1]=c_m[i,:-1]/np.sum(c_m[i,:-1])
            for word in docs:     
                g.append(int(np.where(np.random.multinomial(1, c_m[i])!=0)[0]))
        gibbs[:,j]=g

    word_topic=[]
    for i in range(W):
        word_topic.append(int(Counter(gibbs[i]).most_common(1)[0][0]))
    n_topic=np.max(word_topic)+1
    
    wn_topic = [[] for _ in range(n_topic)]
    wn_doc_topic = [[] for _ in range(n_topic)]

    n = 0
    for i in range(len(corpus)):
        for word in corpus[i]:
            #print(n)
            wn_doc_topic[word_topic[n]].append(word)
            n=n+1
        for j in range(n_topic):
            if wn_doc_topic[j] != []:
                wn_topic[j].append(wn_doc_topic[j])
        wn_doc_topic = [[] for _ in range(n_topic)]        

    wn_topic = [x for x in wn_topic if x != []]
    
    return(wn_topic) 


def hierarchical_LDA(corpus, alpha, beta, lambdas, eta, iters, level,num=5):
    
    from collections import Counter
    import numpy as np
    
    topic = topics(corpus, lambdas)    
    node = [[] for _ in range(level)]
    node_num = [[] for _ in range(level+1)]
    node_num[0].append(1)
    
    word_topic = nodes(corpus, len(topic), alpha, beta, lambdas, eta, iters)
    words = sum(word_topic[0],[])
    node[0].append(words)
    print_word=list(dict(Counter(words).most_common(num)).keys())
  
    temp=word_topic[1:]
    node_num[1].append(len(word_topic[1:]))
    
    for i in range(1,level):
        for j in range(sum(node_num[i])):
            if len(temp)<1:
                break
            word_topic2 = nodes(temp[0], len(topic), alpha, lambdas, eta, iters)
            words2 = sum(word_topic2[0],[])
            node[i].append(words2)
            print_word2=list(dict(Counter(words2).most_common(num)).keys())
           
            temp=temp[1:]
            if len(word_topic2)>2:
                temp.extend(word_topic2[1:])
            node_num[i+1].append(len(word_topic2[1:]))
    return(node,node_num[:level])


def tree_plot(hlda,num=5):
    
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    from collections import Counter
    
    w=hlda[0]
    s=hlda[1]
    graph = pydot.Dot(graph_type='graph')
    for i in range(1,len(s)):
        n1=s[i] # 10
        w1=w[i]
        start=0
        for j in range(len(n1)):
            val=w[i-1][j]
            val=list(dict(Counter(val).most_common(num)).keys())
            root='\n'.join(val)
            n2=n1[j] #8
            end=start+n2
            w2=w1[start:end]
            for k in range(n2):
                w3=w2[k]
                val2=list(dict(Counter(w3).most_common(num)).keys())
                leaf='\n'.join(val2)
                edge = pydot.Edge(root, leaf)
                graph.add_edge(edge)
            start=end
    plt = Image(graph.create_png())
    display(plt)