import numpy as np
import networkx as nx
from gensim.models import Word2Vec

class RandomWalk:
    def __init__(self, graph, p, q, use_rejection_sampling=False):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def simulate_walks(self, num_walks, walk_length, workers, verbose):
        pass

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers):
        self.graph = graph
        self.word2vec = None
        self._embedding = {}
        self.walk = RandomWalk(graph, p = 1, q = 1)
        self.sentences = self.walk.simulate_walks(
            num_walks = num_walks,
            walk_length = walk_length,
            workers = workers,
            verbose = 1
        )
    
    def train(self, window_size=5, embed_size=128, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.word2vec = model
        return model

    def get_embedding(self):
        self._embeddings = {}

        if self.word2vec is None:
            print("model not train")
        else:        
            for node in self.graph.nodes():
                self._embeddings[node] = self.word2vec.wv[node]
        
        return self._embeddings

if __name__ == "__main__":
    G = nx.read_edgelist('../data/wiki.txt',
                        create_using = nx.DiGraph(),
                        nodetype = None,
                        data = [('weight', int)])
    
    model = DeepWalk(G, walk_length = 10, num_walks = 90, workers = 1)
