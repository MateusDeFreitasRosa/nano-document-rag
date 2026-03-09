import struct
import os

class Storage:
    def __init__(self, index_path="index.vlog"):
        self.index_path = index_path

    def save(self, dim, centroids, clusters):
        """
        Saves the index to a binary file.
        Format:
        - Header: dim (I), n_centroids (I), n_vectors (I)
        - Centroids: n_centroids * dim * float32
        - Cluster Index: n_centroids * (offset (Q), count (I))  <- Added for O(1) seek
        - Vectors: n_vectors * dim * float32
        """
        n_centroids = len(centroids)
        # Flatten vectors sorted by cluster
        sorted_vectors = []
        cluster_info = [] # (byte_offset, count)
        
        # We need to calculate offsets first
        # Header size: 4 + 4 + 4 = 12 bytes
        # Centroids size: n_centroids * dim * 4 bytes
        # Cluster Index size: n_centroids * (8 + 4) = 12 bytes per cluster
        
        header_size = 12
        centroids_size = n_centroids * dim * 4
        cluster_index_size = n_centroids * 12
        
        current_vector_offset = header_size + centroids_size + cluster_index_size
        
        for i in range(n_centroids):
            vecs = clusters.get(i, []) # List of (original_idx, vector)
            count = len(vecs)
            cluster_info.append((current_vector_offset, count))
            
            for _, vec in vecs:
                sorted_vectors.extend(vec)
            
            current_vector_offset += count * dim * 4

        n_vectors = len(sorted_vectors) // dim

        with open(self.index_path, "wb") as f:
            # 1. Header
            f.write(struct.pack("III", dim, n_centroids, n_vectors))
            
            # 2. Centroids
            for centroid in centroids:
                f.write(struct.pack(f"{dim}f", *centroid))
            
            # 3. Cluster Index
            for offset, count in cluster_info:
                f.write(struct.pack("QI", offset, count))
            
            # 4. Vectors
            f.write(struct.pack(f"{len(sorted_vectors)}f", *sorted_vectors))

    def load_header(self):
        with open(self.index_path, "rb") as f:
            dim, n_centroids, n_vectors = struct.unpack("III", f.read(12))
        return dim, n_centroids, n_vectors

    def load_centroids(self):
        dim, n_centroids, _ = self.load_header()
        centroids = []
        with open(self.index_path, "rb") as f:
            f.seek(12) # Skip header
            for _ in range(n_centroids):
                vec = struct.unpack(f"{dim}f", f.read(dim * 4))
                centroids.append(list(vec))
        return centroids

    def load_cluster_index(self):
        dim, n_centroids, _ = self.load_header()
        cluster_index = []
        with open(self.index_path, "rb") as f:
            # Skip header + centroids
            offset = 12 + (n_centroids * dim * 4)
            f.seek(offset)
            for _ in range(n_centroids):
                offset, count = struct.unpack("QI", f.read(12))
                cluster_index.append((offset, count))
        return cluster_index

    def load_cluster_vectors(self, cluster_id, cluster_index_entry, dim):
        """
        Loads vectors for a specific cluster using the index entry (offset, count).
        """
        offset, count = cluster_index_entry
        if count == 0:
            return []
            
        vectors = []
        with open(self.index_path, "rb") as f:
            f.seek(offset)
            # Read all vectors for this cluster at once
            data = f.read(count * dim * 4)
            # Unpack
            all_floats = struct.unpack(f"{count * dim}f", data)
            
            # Reshape
            for i in range(count):
                start = i * dim
                end = start + dim
                vectors.append(list(all_floats[start:end]))
        return vectors
