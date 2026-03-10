import os
import json
from .math_ops import cosine_similarity
from .clusterizer import Clusterizer
from .storage import Storage

class NanoRAG:
    def __init__(self, document_id, storage_dir=".", n_clusters=None):
        """
        Inicializa o NanoRAG para um documento específico.
        
        Args:
            document_id: Identificador único do documento (ex: 'manual_tecnico').
                         Isso gerará '{document_id}.vlog' e '{document_id}.json'.
            storage_dir: Diretório onde os arquivos de índice serão armazenados.
            n_clusters: Quantidade de clusters (K). Se None, será calculado automaticamente como sqrt(N).
        """
        self.document_id = document_id
        self.storage_dir = storage_dir
        self.n_clusters = n_clusters
        
        if storage_dir != "." and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.index_path = os.path.join(storage_dir, f"{document_id}.vlog")
        self.metadata_path = os.path.join(storage_dir, f"{document_id}.json")
        
        self.storage = Storage(self.index_path)
        # O K será definido dinamicamente no momento da indexação se n_clusters for None
        initial_k = n_clusters if n_clusters is not None else 1
        self.clusterizer = Clusterizer(k=initial_k)
        self.metadata = {}

    def index(self, embeddings, contents, metadatas=None):
        """
        Gera o índice vetorial para este documento com cálculo automático de clusters.
        
        Args:
            embeddings: Lista de vetores (List[List[float]]) usados para a busca.
            contents: Lista de strings (List[str]) que representam o chunk.
            metadatas: Lista opcional de dicionários de metadados (List[dict]).
        """
        if not embeddings:
            return

        # Cálculo automático do número de clusters (K)
        # Heurística: K = sqrt(N), onde N é o número de chunks
        if self.n_clusters is None:
            import math
            n_samples = len(embeddings)
            # Garante pelo menos 1 cluster e no máximo o número de amostras
            k_auto = int(math.sqrt(n_samples))
            self.clusterizer.k = max(1, k_auto)
        else:
            self.clusterizer.k = self.n_clusters

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
        """Método interno para construir e salvar o índice do documento."""
        if not embeddings:
            return

        dim = len(embeddings[0])
        print(f"Documento: {self.document_id} | Dimensão: {dim}")

        # 1. Cluster
        print(f"Clusterizando {len(embeddings)} vetores...")
        centroids, clusters = self.clusterizer.fit(embeddings)
        
        # 2. Save Index (Binary)
        print("Salvando índice binário (.vlog)...")
        
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
        print("Salvando metadados (.json)...")
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(ordered_metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Indexação do documento '{self.document_id}' concluída.")

    def search(self, query_vector, top_k=3, margin_chars=0):
        """
        Busca os trechos mais similares dentro deste documento.
        
        Args:
            query_vector: Vetor da query (List[float]).
            top_k: Número de resultados.
            margin_chars: Quantidade de caracteres para expandir (vizinhos).
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
                content = meta["content"]
                
                # Expansão Dinâmica de Contexto (Vizinhos)
                if margin_chars > 0:
                    prev_meta = self.metadata.get(global_idx - 1)
                    next_meta = self.metadata.get(global_idx + 1)
                    
                    prefix = prev_meta["content"][-margin_chars:] if prev_meta else ""
                    suffix = next_meta["content"][:margin_chars] if next_meta else ""
                    content = f"{prefix}{content}{suffix}"

                final_output.append({
                    "score": score,
                    "content": content,
                    "metadata": meta["metadata"],
                    "cluster_id": meta["cluster_id"]
                })
                
        return final_output
