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
**Ouput:** matrix of vertex representations $\Phi \in \mathbb{R}^{|V| \times d}$   
1. Initialization: Sample $\Phi$ from $U^{|V| \times d} $   
2. Build a binary Tree $T$ from $V$   
3. **for** $i$ = 0 to $\gamma$ **do**   
4. $\qquad$ $O$ = Shuffle($V$)   
5. $\qquad$ **for each** $v_i \in O$ **do**   
6. $\qquad\qquad$ $W_{v_i} = RandomWalk(G, v_i, t)$   
7. $\qquad\qquad$ $SkipGram(\Phi, W_{v_i}, w)$   
8. $\qquad$ **end for**   
9. **end for**   
