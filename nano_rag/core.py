import os
import json
from .math_ops import cosine_similarity
from .clusterizer import Clusterizer
from .storage import Storage

class NanoRAG:
    def __init__(self, db_name="vector_db"):
        """
        Inicializa o NanoRAG.
        
        Args:
            db_name: Nome base para os arquivos de banco de dados (ex: 'meu_banco').
                     Isso gerará 'meu_banco.vlog' e 'meu_banco.json'.
        """
        self.db_name = db_name
        self.index_path = f"{db_name}.vlog"
        self.metadata_path = f"{db_name}.json"
        
        self.storage = Storage(self.index_path)
        self.clusterizer = Clusterizer(k=10) # Default K, can be tuned
        self.metadata = {}

    def insert(self, documents):
        """
        Insere uma lista de documentos já processados.
        
        Args:
            documents: Lista de dicts, onde cada dict tem:
                - 'embedding': List[float]
                - 'content': str
                - 'metadata': dict (opcional)
        """
        if not documents:
            print("Nenhum documento fornecido.")
            return

        embeddings = [doc['embedding'] for doc in documents]
        self._build_index(embeddings, documents)

    def fit(self, embeddings, contents, metadatas=None):
        """
        Método facilitador para indexação em lote.
        
        Args:
            embeddings: Lista de vetores (List[List[float]])
            contents: Lista de strings (List[str])
            metadatas: Lista opcional de dicionários de metadados (List[dict])
        """
        if len(embeddings) != len(contents):
            raise ValueError("As listas de embeddings e contents devem ter o mesmo tamanho.")
        
        if metadatas and len(metadatas) != len(embeddings):
            raise ValueError("A lista de metadatas deve ter o mesmo tamanho das outras, se fornecida.")

        documents = []
        for i in range(len(embeddings)):
            doc = {
                "embedding": embeddings[i],
                "content": contents[i],
                "metadata": metadatas[i] if metadatas else {}
            }
            documents.append(doc)
        
        self._build_index(embeddings, documents)

    def _build_index(self, embeddings, documents):
        """Internal method to build and save the index."""
        if not embeddings:
            return

        dim = len(embeddings[0])
        print(f"Dimensão do embedding: {dim}")

        # 1. Cluster
        print("Clusterizando...")
        centroids, clusters = self.clusterizer.fit(embeddings)
        
        # 2. Save Index (Binary)
        print("Salvando índice binário...")
        
        ordered_metadata = {}
        current_idx = 0
        
        for i in range(len(centroids)):
            cluster_vectors = clusters.get(i, [])
            for original_idx, _ in cluster_vectors:
                doc = documents[original_idx]
                ordered_metadata[current_idx] = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "cluster_id": i
                }
                current_idx += 1

        self.storage.save(dim, centroids, clusters)

        # 3. Save Metadata
        print("Salvando metadados...")
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(ordered_metadata, f, ensure_ascii=False, indent=2)
            
        print("Indexação concluída.")

    def query(self, query_vector, top_k=3):
        """
        Performs the 2-step search:
        1. Find closest centroid.
        2. Search vectors within that cluster.
        
        Args:
            query_vector: List[float] representing the query embedding.
            top_k: Number of results to return.
        """
        # 1. Load Centroids
        centroids = self.storage.load_centroids()
        if not centroids:
            return []

        # 2. Find Best Cluster
        best_cluster_id = -1
        max_sim = -1.0
        
        # Using Cosine Similarity for query vs centroids
        for i, centroid in enumerate(centroids):
            sim = cosine_similarity(query_vector, centroid)
            if sim > max_sim:
                max_sim = sim
                best_cluster_id = i
        
        if best_cluster_id == -1:
            return []

        # 3. Load Vectors for that Cluster
        # We need the cluster index first
        cluster_index = self.storage.load_cluster_index() # List of (offset, count)
        dim = len(query_vector)
        
        cluster_vectors = self.storage.load_cluster_vectors(best_cluster_id, cluster_index[best_cluster_id], dim)
        
        # 4. Find Best Vectors in Cluster
        results = []
        for i, vec in enumerate(cluster_vectors):
            score = cosine_similarity(query_vector, vec)
            results.append((score, i)) # i is the index WITHIN the cluster
        
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:top_k]
        
        # 5. Retrieve Metadata
        # We need to map the cluster-local index back to the global storage index
        # The global index is: sum(counts of previous clusters) + local_index
        
        global_offset = 0
        for i in range(best_cluster_id):
            global_offset += cluster_index[i][1] # count
            
        final_output = []
        
        # Load metadata if not loaded
        if not self.metadata:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    # JSON keys are strings, convert to int
                    data = json.load(f)
                    self.metadata = {int(k): v for k, v in data.items()}
            else:
                return []

        for score, local_idx in top_results:
            global_idx = global_offset + local_idx
            meta = self.metadata.get(global_idx)
            if meta:
                final_output.append({
                    "score": score,
                    "content": meta["content"],
                    "metadata": meta["metadata"],
                    "cluster_id": meta["cluster_id"]
                })
                
        return final_output
