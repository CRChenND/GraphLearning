# Graph embedding
Intuitively, we can use one-hot encoding or adjacency matrix to represent every node in a graph. But it has two disadvantages:
1. It results in a very **sparse** graph when there are many nodes.
2. It losts node features or extra information in the graph.

Therefore, we want to apply graph embedding to obtain a low-dimensional continuous representation to keep more information in the graph.

## [DeepWalk](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)
​​DeepWalk originates from word2vec, which uses the co-occurrence relationship between nodes to learn the representation. To describe the co-occurrence relationship between nodes, DeepWalk uses *truncated random walk* (random walk with fixed length) to sample nodes in the graph.

**Algorithm** $ DeepWalk(G, w, d, \gamma, t) $   
**Input:** graph $G(V,E)$   
$\qquad$ window size $w$   
$\qquad$ embedding size $d$    
$\qquad$ walks per vertex $\gamma$   
$\qquad$ walk length $t$   
**Ouput:** matrix of vertex representations $\Phi \in \mathbb{R}^{|V| \times d}$
1. Initialization: Sample $\Phi$ from $U^{|V| \times d} $   
2. Build a binary Tree $T$ from $V$  <font color=red>&larr Hierarchical Softmax</font>
3. **for** $i$ = 0 to $\gamma$ **do**   
4. $\qquad$ $O = Shuffle(V)$   
5. $\qquad$ **for each** $v_i \in O$ **do**   
6. $\qquad\qquad$ $W_{v_i} = RandomWalk(G, v_i, t)$   
7. $\qquad\qquad$ $SkipGram(\Phi, W_{v_i}, w)$   <font color=red>$\larr$ $\underset {\Phi}{minimize}\ -log\ Pr(\{v_{i-w},...,v_{i-1},v_{i+1},v_{i+w}\}\ |\ \Phi(v_i))$</font>
8. $\qquad$ **end for**   
9. **end for**   

### The advantage of using random walk:
1. **Parallelizable**: For a large network, random walks can be performed at different vertices at the same time, reducing the sampling time.
2. **Scalable**: The evolution of the network is usually the change of local points and edges. Such changes will only affect some random walk paths, so it is not necessary to recalculate the random walk of the entire network every time during the evolution of the network.

### Why use word2vec:
The distribution of random walks in the network is similar to the power-law distribution of the sequence of sentences in NLP in the corpus.

### Why use skip-gram:
1. Use missing words(nodes) to predict context because it is too complicated to calculate the context.
1. Consider bilateral nodes within the window size $w$ for the given node.
2. Insensitive to the order of nodes.
