import struct
import os
import json

class Storage:
    def __init__(self, index_path, s3_client=None, s3_bucket=None):
        self.index_path = index_path
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.pos = 0

    def _read_bytes(self, offset, size):
        """Lê uma quantidade específica de bytes, seja do S3 ou do disco local."""
        if self.s3_client and self.s3_bucket:
            # S3 Range Request: baixa apenas o pedaço necessário
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=self.index_path,
                Range=f"bytes={offset}-{offset + size - 1}"
            )
            return response['Body'].read()
        else:
            # Leitura de arquivo local
            with open(self.index_path, "rb") as f:
                f.seek(offset)
                return f.read(size)

    def save(self, dim, centroids, clusters):
        """Saves the index to a binary file (Local only)."""
        n_centroids = len(centroids)
        sorted_vectors = []
        cluster_info = [] # (byte_offset, count)
        
        header_size = 12
        centroids_size = n_centroids * dim * 4
        cluster_index_size = n_centroids * 12
        
        current_vector_offset = header_size + centroids_size + cluster_index_size
        
        for i in range(n_centroids):
            vecs = clusters.get(i, [])
            count = len(vecs)
            cluster_info.append((current_vector_offset, count))
            
            for _, vec in vecs:
                sorted_vectors.extend(vec)
            
            current_vector_offset += count * dim * 4

        n_vectors = len(sorted_vectors) // dim

        with open(self.index_path, "wb") as f:
            f.write(struct.pack("III", dim, n_centroids, n_vectors))
            for centroid in centroids:
                f.write(struct.pack(f"{dim}f", *centroid))
            for offset, count in cluster_info:
                f.write(struct.pack("QI", offset, count))
            f.write(struct.pack(f"{len(sorted_vectors)}f", *sorted_vectors))

    def load_header(self):
        data = self._read_bytes(0, 12)
        dim, n_centroids, n_vectors = struct.unpack("III", data)
        return dim, n_centroids, n_vectors

    def load_centroids(self):
        dim, n_centroids, _ = self.load_header()
        centroids = []
        # Lê todos os centróides de uma vez para otimizar I/O
        size = n_centroids * dim * 4
        data = self._read_bytes(12, size)
        
        for i in range(n_centroids):
            start = i * dim * 4
            end = start + (dim * 4)
            vec = struct.unpack(f"{dim}f", data[start:end])
            centroids.append(list(vec))
        return centroids

    def load_cluster_index(self):
        dim, n_centroids, _ = self.load_header()
        cluster_index = []
        offset = 12 + (n_centroids * dim * 4)
        size = n_centroids * 12
        data = self._read_bytes(offset, size)
        
        for i in range(n_centroids):
            start = i * 12
            end = start + 12
            c_offset, count = struct.unpack("QI", data[start:end])
            cluster_index.append((c_offset, count))
        return cluster_index

    def load_cluster_vectors(self, cluster_id, cluster_index_entry, dim):
        offset, count = cluster_index_entry
        if count == 0:
            return []
            
        size = count * dim * 4
        data = self._read_bytes(offset, size)
        all_floats = struct.unpack(f"{count * dim}f", data)
        
        vectors = []
        for i in range(count):
            start = i * dim
            end = start + dim
            vectors.append(list(all_floats[start:end]))
        return vectors
