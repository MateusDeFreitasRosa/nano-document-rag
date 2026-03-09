import random
import math
from .math_ops import euclidean_distance, vector_add, vector_scale, magnitude

class Clusterizer:
    def __init__(self, k=10, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = []
        self.clusters = {}  # Map: centroid_index -> list of vector_indices

    def fit(self, vectors):
        """
        Runs K-Means clustering on the provided vectors.
        Args:
            vectors: List of vectors (lists of floats).
        Returns:
            centroids: List of centroid vectors.
            clusters: Dict mapping cluster_id to list of (vector_index, vector).
        """
        if not vectors:
            return [], {}
        
        n_samples = len(vectors)
        if n_samples < self.k:
            self.k = n_samples

        # 1. Initialize Centroids (Randomly pick k vectors)
        # Using a fixed seed for reproducibility during dev, remove for prod if needed
        # random.seed(42) 
        self.centroids = random.sample(vectors, self.k)

        for iteration in range(self.max_iters):
            # 2. Assign vectors to nearest centroid
            new_clusters = {i: [] for i in range(self.k)}
            
            # Keep track of vector indices to map back later
            for vec_idx, vec in enumerate(vectors):
                best_centroid_idx = -1
                min_dist = float('inf')
                
                for cent_idx, centroid in enumerate(self.centroids):
                    dist = euclidean_distance(vec, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_centroid_idx = cent_idx
                
                if best_centroid_idx != -1:
                    new_clusters[best_centroid_idx].append((vec_idx, vec))

            # 3. Update Centroids
            new_centroids = []
            max_shift = 0.0
            
            for i in range(self.k):
                cluster_vectors = [v for _, v in new_clusters[i]]
                if not cluster_vectors:
                    # Handle empty cluster: keep old centroid or re-initialize
                    new_centroids.append(self.centroids[i])
                    continue
                
                # Calculate mean vector
                dim = len(cluster_vectors[0])
                sum_vec = [0.0] * dim
                for v in cluster_vectors:
                    sum_vec = vector_add(sum_vec, v)
                
                mean_vec = vector_scale(sum_vec, 1.0 / len(cluster_vectors))
                new_centroids.append(mean_vec)

                # Check shift
                shift = euclidean_distance(self.centroids[i], mean_vec)
                if shift > max_shift:
                    max_shift = shift

            self.centroids = new_centroids
            self.clusters = new_clusters

            # 4. Check Convergence
            if max_shift < self.tolerance:
                break
        
        return self.centroids, self.clusters
