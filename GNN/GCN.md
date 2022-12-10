# [Graph convolutional networks (ICLR '17)](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89)

### What is graph neural network (GNN):
GNN is a genre of neural networks dedicated to processing graph data. Specifically, given the adjacency matrix $A \in \mathbb{R}^{N \times N}$ and the matrix of activations $H^{(l)} \in \mathbb{R}^{N \times D}$ in the $l^{th}$ layer, we will try to learn a mapping function $f$ to obtain the node feature representation for the next layer.

$$
H^{(l+1)} = f(A, H^{(l)}),
$$

Generally, the mapping function $f$ would use the adjacency matrix $A$ to aggregate some features from the nodes and their neighbors to generate the new feature representation. The reason behinds this operation is that a node feature is determined by not only the nodes themselves (e.g., users in social media) and their neighbors (e.g., users' friends).

**Therefore, the objective for all GNNs is to design their mapping function $f$.**

### How does GCN design its mapping function $f$:

$$
H^{(l+1)} = \sigma(\hat{D}^{- \frac{1}{2}}\hat{A}\hat{D}^{\frac{1}{2}}H^{(l)}W^{(l)}),
$$

$$
\hat{A}=A+I_N
$$

$$
\hat{D}=\sum_{j} \hat{A_{ij}} = D + I_N
$$

$\hat{A}$ is the adjacency matrix of graph $G$ with added self-connections,               
$I_N$ is identity matrix,                 
$\hat{D}$ is a degree matrix,
$W^{(l)}$ is a layer-specific trainable weight matrix,            
$\sigma$ is an activation function, such as the $ReLU$.

### Why does GCN design such a function $f$:

### Prerequisite 1: spectral graph theory
Spectral graph theory is the study of the properties of a graph. That is, it limits the study of matrix properties in linear algebra to the adjacency matrix of graphs. So spectral graph theory is a sub area of linear algebra.

#### Recap linear algebra

$$
A_1 \vec{x} = \lambda \vec{x}
$$

For a matrix $A_1 \in \mathbb{R^{N \times N}}$
- **Eigenvector** $\vec{x}$: a non-zero vector that, when a linear transformation is applied to it, does not change its direction. 
- **Eigenvalue** $\lambda$: a scalar value that represents the amount by which the eigenvector is scaled or stretched when the transformation is applied.

$$
A_2 = U \Lambda U^T \\
UU^T = I\\
\Lambda = 
\begin{bmatrix}
\lambda_1 & & &\\
& \lambda_2 & &\\
& & \ddots & \\
& & & \lambda_N
\end{bmatrix}
$$

For a **real symmetrical matrix** $A_2$, 
- all of its eigenvalues $\Lambda$ are real
- all of its eigenvectors are orthogonal to each other 

If all of the eigenvalues of $A_2$ are non-negative, then we call it **semi-positive defined matrix**.


For a matrix $A \in \mathbb{R^{N \times N}}$, the quadratic form of vector $\vec{x}$ is
$$
\vec{x}^TA\vec{x}
$$

The **Rayleigh quotient** measures how well a given vector or matrix fits a particular linear operator or function. In general, the Rayleigh quotient is defined as 

$$
\frac{\vec{x}^TA\vec{x}}{\vec{x}^T\vec{x}}
$$

if $\vec{x}$ is the eigenvector of matrix $A$, then we can prove its Rayleigh quotient equals to the eigenvalue $\lambda$ of matrix $A$:

$$
\frac{\vec{x}^TA\vec{x}}{\vec{x}^T\vec{x}} = 
\frac{\vec{x}^T (\lambda\vec{x})}{\vec{x}^T\vec{x}} = 
\frac{\lambda(\vec{x}^T\vec{x})}{\vec{x}^T\vec{x}} =
\lambda
$$

#### Properties of matrices related to $A$ in spectral graph theory

- Laplacian matrix: $L = D-A$
- Symmetrical normalization of the Laplacian matrix: $L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$

Both of the matrices are real symmetrical matrix.

We can use Rayleigh quotient to prove both of them are also semi-positive defined, that is, to prove the eigenvalue $\Lambda$ of Laplacian matrix $L$ and the eigenvalue $\Lambda_{sym}$ of symmetrical normalization of the Laplacian matrix: $L_{sym} is non-negative:

$$
\Lambda = \frac{\vec{x}^TL\vec{x}}{\vec{x}^T\vec{x}} \geq 0\\
$$

$$
\Lambda_{sym} = \frac{\vec{x}^TL_{sym}\vec{x}}{\vec{x}^T\vec{x}} \geq 0
$$

We can construct a matrix $G_{(i,j)}$, where the element in the $i^{th}$ row, $i^{th}$ column and the $j^{th}$ row, $j^{th}$ column are 1, while the element in the $i^{th}$ row, $j^{th}$ column and the $j^{th}$ row, $i^{th}$ column are -1,

$$
G_{(i,j)} = 
\begin{bmatrix}
\ddots & & & &\\
& 1_{(i,i)} &\cdots &-1_{(i,j)} &\\
& \vdots & \ddots & \vdots &\\
& -1_{(j,i)} &\cdots & 1_{(j,j)} & \\
& & & & \ddots \\
\end{bmatrix}\\
\vec{x}^TG_{(i,j)}\vec{x} = \vec{x}^T \begin{bmatrix}
\vdots\\
x_i - x_j\\
\vdots\\
x_j - x_i\\
\vdots
\end{bmatrix} = x_i(x_i-x_j)+x_j(x_j-x_i) = (x_i - x_j)^2
$$ 

$$
L = D-A = \sum_{(i,j) \in E} G_{(i,j)}\\
$$

$$
\vec{x}^TL\vec{x} = \vec{x}^T(\sum_{(i,j) \in E} G_{(i,j)})\vec{x} = \sum_{(i,j) \in E} \vec{x}^TG_{(i,j)}\vec{x} = \sum_{(i,j) \in E} (x_i - x_j)^2 \geq 0\\
$$

$$
\vec{x}^TL_{sym}\vec{x} = (\vec{x}^TD^{-\frac{1}{2}})L(D^{-\frac{1}{2}}\vec{x}) = \sum_{(i,j) \in E} (\frac{x_i}{\sqrt{x_i}} - \frac{x_j}{\sqrt{x_j}})^2 \geq 0
$$

$$
\because \vec{x}^T\vec{x} > 0
$$

$$
\frac{\vec{x}^TL_{sym}\vec{x}}{\vec{x}^T\vec{x}} \geq 0
$$

$$
\Lambda_{sym} \geq 0
$$

**We can further prove that $\Lambda_{sym} \in [0,2]$**,

We can construct a new matrix $G^{pos}_{(i,j)}$,
$$
G^{pos}_{(i,j)} = 
\begin{bmatrix}
\ddots & & & &\\
& 1_{(i,i)} &\cdots &1_{(i,j)} &\\
& \vdots& \ddots &\vdots &\\
& 1_{(j,i)} &\cdots & 1_{(j,j)} & \\
& & & & \ddots \\
\end{bmatrix}\\
\vec{x}^TG_{(i,j)}\vec{x} = \vec{x}^T \begin{bmatrix}
\vdots\\
x_i + x_j\\
\vdots\\
x_j + x_i\\
\vdots
\end{bmatrix} = x_i(x_i+x_j)+x_j(x_j+x_i) = (x_i + x_j)^2
$$ 

$$
L^{pos} = D+A = \sum_{(i,j) \in E} G^{pos}_{(i,j)}
$$

$$
L^{pos}_{sym} = D^{-\frac{1}{2}}L^{pos}D^{-\frac{1}{2}} = I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
$$

$$
\vec{x}^TL^{pos}\vec{x} = \sum_{(i,j) \in E} (x_i + x_j)^2
$$

$$
\vec{x}^TL^{pos}_{sym}\vec{x} = \sum_{(i,j) \in E} (\frac{x_i}{\sqrt{x_i}} + \frac{x_j}{\sqrt{x_j}})^2 \geq 0
$$

$$
\vec{x}^TL^{pos}_{sym}\vec{x} = \vec{x}^T(I + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})\vec{x}
= \vec{x}^T\vec{x} + \vec{x}^TD^{-\frac{1}{2}}AD^{-\frac{1}{2}}\vec{x} \geq 0 
$$

$$
\vec{x}^T\vec{x} \geq -\vec{x}^TD^{-\frac{1}{2}}AD^{-\frac{1}{2}}\vec{x}
$$

$$
2\vec{x}^T\vec{x} \geq \vec{x}^T\vec{x}-\vec{x}^TD^{-\frac{1}{2}}AD^{-\frac{1}{2}}\vec{x}
$$

$$
2\vec{x}^T\vec{x} \geq \vec{x}^T(I-D^{-\frac{1}{2}}AD^{-\frac{1}{2}})\vec{x}
$$

$$
2\vec{x}^T\vec{x} \geq \vec{x}^TL_{sym}\vec{x}
$$

$$
\frac{\vec{x}^TL_{sym}\vec{x}}{\vec{x}^T\vec{x}} \leq 2
$$

$$
\Lambda_{sym} \leq 2
$$


### Prerequisite 2: Fourier transformation

**[noted]** GCN only considers undirected simple graph (contains no duplicate edges and no loops)