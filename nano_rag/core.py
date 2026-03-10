import os
import json
from .math_ops import cosine_similarity
from .clusterizer import Clusterizer
from .storage import Storage

class NanoDocumentRAG:
    def __init__(self, document_id, storage_dir=".", n_clusters=None, s3_client=None, s3_bucket=None):
        """
        Inicializa o NanoDocumentRAG para um documento específico.
        
        Args:
            document_id: Identificador único do documento (ex: 'manual_tecnico').
            storage_dir: Diretório local ou prefixo no S3.
            n_clusters: Quantidade de clusters (K).
            s3_client: Objeto boto3.client('s3') injetado (opcional).
            s3_bucket: Nome do bucket no S3 (necessário se s3_client for usado).
        """
        self.document_id = document_id
        self.storage_dir = storage_dir
        self.n_clusters = n_clusters
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        
        # Caminhos físicos
        if s3_client and s3_bucket:
            # No S3, usamos caminhos com /
            self.index_path = f"{storage_dir}/{document_id}.vlog".lstrip("./")
            self.metadata_path = f"{storage_dir}/{document_id}.json".lstrip("./")
        else:
            # Localmente, usamos os caminhos do SO
            if storage_dir != "." and not os.path.exists(storage_dir):
                os.makedirs(storage_dir)
            self.index_path = os.path.join(storage_dir, f"{document_id}.vlog")
            self.metadata_path = os.path.join(storage_dir, f"{document_id}.json")
        
        self.storage = Storage(self.index_path, s3_client=s3_client, s3_bucket=s3_bucket)
        initial_k = n_clusters if n_clusters is not None else 1
        self.clusterizer = Clusterizer(k=initial_k)
        self.metadata = {}

    def _load_metadata(self):
        """Carrega os metadados do disco local ou do S3."""
        if self.s3_client and self.s3_bucket:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=self.metadata_path)
            data = json.loads(response['Body'].read().decode('utf-8'))
        else:
            if not os.path.exists(self.metadata_path):
                return False
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        self.metadata = {int(k): v for k, v in data.items()}
        return True

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
            
        # 4. Upload Automático para S3 (Se configurado)
        if self.s3_bucket:
            try:
                import boto3
                s3 = self.s3_client or boto3.client('s3')
                print(f"📤 Fazendo upload automático para S3: s3://{self.s3_bucket}/{self.index_path}")
                
                # Normaliza caminhos para o S3 (sempre usa /)
                s3_index_key = self.index_path.replace("\\", "/")
                s3_meta_key = self.metadata_path.replace("\\", "/")
                
                s3.upload_file(self.index_path, self.s3_bucket, s3_index_key)
                s3.upload_file(self.metadata_path, self.s3_bucket, s3_meta_key)
                print("✅ Upload para S3 concluído com sucesso!")
            except ImportError:
                print("⚠️ Aviso: 'boto3' não encontrado. O índice foi salvo localmente, mas não pôde ser enviado ao S3.")
                print("   Instale com: pip install 'nano-document-rag[aws]'")
            except Exception as e:
                print(f"❌ Erro ao fazer upload para o S3: {str(e)}")

        print(f"Indexação do documento '{self.document_id}' concluída.")

    def search(self, query_vector, top_k=3, margin_chars=0):
        """
        Busca os trechos mais similares dentro deste documento.
        """
        # 1. Load Centroids
        centroids = self.storage.load_centroids()
        if not centroids:
            return []

        # 2. Find Best Cluster
        best_cluster_id = -1
        max_sim = -1.0
        for i, centroid in enumerate(centroids):
            sim = cosine_similarity(query_vector, centroid)
            if sim > max_sim:
                max_sim = sim
                best_cluster_id = i
        
        if best_cluster_id == -1:
            return []

        # 3. Load Cluster Vectors
        cluster_index = self.storage.load_cluster_index()
        dim = len(query_vector)
        cluster_vectors = self.storage.load_cluster_vectors(best_cluster_id, cluster_index[best_cluster_id], dim)
        
        # 4. Search in Cluster
        results = []
        for i, vec in enumerate(cluster_vectors):
            score = cosine_similarity(query_vector, vec)
            results.append((score, i))
        
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:top_k]
        
        # 5. Retrieve Metadata
        global_offset = sum(cluster_index[i][1] for i in range(best_cluster_id))
            
        if not self.metadata:
            if not self._load_metadata():
                return []

        final_output = []
        for score, local_idx in top_results:
            global_idx = global_offset + local_idx
            meta = self.metadata.get(global_idx)
            if meta:
                content = meta["content"]
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
