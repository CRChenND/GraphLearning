# Why should we use graph embedding?
Intuitively, we can use one-hot encoding or adjacency matrix to represent every node in a graph. But it has two disadvantages:
1. It results in a very **sparse** graph when there are many nodes.
2. It losts node features or extra information in the graph.

Therefore, we want to apply graph embedding to obtain a more condensed representation to keep more information in the graph.

# [DeepWalk](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)
​​DeepWalk originates from word2vec, which uses the co-occurrence relationship between nodes to learn the representation. To describe the co-occurrence relationship between nodes, DeepWalk uses random walk to sample nodes in the graph.

**Algorithm** $ DeepWalk(G, w, d, \gamma, t) $
**Input:** graph $G(V,E)$
$\qquad$ window size $w$
$\qquad$ embedding size $d$ 
$\qquad$ walks per vertex $\gamma$
$\qquad$ walk length $t$
**Ouput:** matrix of vertex representations $\Phi \in \R^{|V| \times d}$
