import os
import json
import math
from .math_ops import cosine_similarity
from .clusterizer import Clusterizer
from .storage import Storage

class NanoDocumentRAG:
    def __init__(self, storage_dir=".", s3_client=None, s3_bucket=None):
        """
        Gerenciador de documentos vetoriais NanoDocumentRAG.
        
        Args:
            storage_dir: Pasta local ou prefixo no S3 para armazenar os índices.
            s3_client: Cliente boto3 (opcional).
            s3_bucket: Nome do bucket no S3 (opcional).
        """
        self.storage_dir = storage_dir
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        
        # Se for local, garante que a pasta existe
        if not s3_client and storage_dir != "." and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def _get_paths(self, document_id):
        """Gera os caminhos físicos (locais ou S3) para um documento."""
        if self.s3_client and self.s3_bucket:
            # No S3, usamos caminhos com / e sem o ./ inicial
            vlog = f"{self.storage_dir}/{document_id}.vlog".lstrip("./").replace("\\", "/")
            meta = f"{self.storage_dir}/{document_id}.json".lstrip("./").replace("\\", "/")
            return vlog, meta
        else:
            vlog = os.path.join(self.storage_dir, f"{document_id}.vlog")
            meta = os.path.join(self.storage_dir, f"{document_id}.json")
            return vlog, meta

    def create_or_replace_document(self, document_id, embeddings, contents, metadatas=None, n_clusters=None):
        """
        Cria um novo índice para o documento ou sobrescreve o existente.
        """
        self._build_and_save(document_id, embeddings, contents, metadatas, n_clusters)

    def add_to_document(self, document_id, embeddings, contents, metadatas=None, n_clusters=None):
        """
        Adiciona novos vetores a um documento existente (Append com re-indexação).
        """
        vlog_path, meta_path = self._get_paths(document_id)
        
        existing_embeddings = []
        existing_contents = []
        existing_metadatas = []
        
        # Tenta carregar dados existentes para fazer o merge
        try:
            # Precisamos baixar temporariamente para ler se estiver no S3
            temp_vlog = vlog_path
            temp_meta = meta_path
            
            if self.s3_client:
                temp_vlog = f"temp_{document_id}.vlog"
                temp_meta = f"temp_{document_id}.json"
                self.s3_client.download_file(self.s3_bucket, vlog_path, temp_vlog)
                self.s3_client.download_file(self.s3_bucket, meta_path, temp_meta)

            # Carrega metadados
            with open(temp_meta, "r", encoding="utf-8") as f:
                old_meta_dict = json.load(f)
            
            # Carrega vetores usando o Storage
            storage = Storage(temp_vlog)
            dim, _, _ = storage.load_header()
            cluster_idx = storage.load_cluster_index()
            
            for i in range(len(cluster_idx)):
                vecs = storage.load_cluster_vectors(i, cluster_idx[i], dim)
                existing_embeddings.extend(vecs)
            
            # Ordena metadados antigos para extrair conteúdo e metadados originais
            for i in range(len(old_meta_dict)):
                item = old_meta_dict[str(i)]
                existing_contents.append(item["content"])
                existing_metadatas.append(item["metadata"])
                
            # Limpa arquivos temporários se necessário
            if self.s3_client:
                os.remove(temp_vlog)
                os.remove(temp_meta)
                
        except Exception as e:
            print(f"ℹ️ Documento '{document_id}' não encontrado ou erro ao carregar. Iniciando novo: {e}")

        # Merge dos dados novos com os antigos
        all_embeddings = existing_embeddings + embeddings
        all_contents = existing_contents + contents
        all_metadatas = existing_metadatas + (metadatas if metadatas else [{}] * len(contents))
        
        self._build_and_save(document_id, all_embeddings, all_contents, all_metadatas, n_clusters)

    def _build_and_save(self, document_id, embeddings, contents, metadatas, n_clusters):
        """Lógica central de clusterização e persistência."""
        vlog_path, meta_path = self._get_paths(document_id)
        
        # 1. Cálculo de Clusters
        n_samples = len(embeddings)
        k = n_clusters if n_clusters else max(1, int(math.sqrt(n_samples)))
        clusterizer = Clusterizer(k=k)
        centroids, clusters = clusterizer.fit(embeddings)
        
        # 2. Preparação de Metadados Ordenados por Cluster
        ordered_meta = {}
        curr = 0
        for i in range(len(centroids)):
            for orig_idx, _ in clusters.get(i, []):
                ordered_meta[curr] = {
                    "content": contents[orig_idx],
                    "metadata": metadatas[orig_idx] if metadatas else {},
                    "cluster_id": i
                }
                curr += 1
        
        # 3. Salvamento (Sempre gera local primeiro)
        # Se for S3, usamos arquivos temporários para o upload
        local_vlog = vlog_path
        local_meta = meta_path
        if self.s3_client:
            local_vlog = f"upload_{document_id}.vlog"
            local_meta = f"upload_{document_id}.json"

        storage = Storage(local_vlog)
        storage.save(len(embeddings[0]), centroids, clusters)
        with open(local_meta, "w", encoding="utf-8") as f:
            json.dump(ordered_meta, f, ensure_ascii=False, indent=2)
            
        # 4. Sincronização com S3
        if self.s3_client and self.s3_bucket:
            self.s3_client.upload_file(local_vlog, self.s3_bucket, vlog_path)
            self.s3_client.upload_file(local_meta, self.s3_bucket, meta_path)
            os.remove(local_vlog)
            os.remove(local_meta)
            print(f"✅ Documento '{document_id}' sincronizado com S3.")
        else:
            print(f"✅ Documento '{document_id}' salvo localmente.")

    def search(self, document_id, query_vector, top_k=3, margin_chars=0):
        """
        Busca os trechos mais similares dentro de um documento específico.
        """
        vlog_path, meta_path = self._get_paths(document_id)
        
        # Inicializa Storage (Modo Local ou S3 Range Request)
        storage = Storage(vlog_path, s3_client=self.s3_client, s3_bucket=self.s3_bucket)
        
        # 1. Busca Centróide
        centroids = storage.load_centroids()
        if not centroids: return []
        
        best_id = -1
        max_sim = -1.0
        for i, c in enumerate(centroids):
            sim = cosine_similarity(query_vector, c)
            if sim > max_sim:
                max_sim = sim
                best_id = i
        
        if best_id == -1: return []
        
        # 2. Busca no Cluster
        c_idx = storage.load_cluster_index()
        dim = len(query_vector)
        vecs = storage.load_cluster_vectors(best_id, c_idx[best_id], dim)
        
        res = []
        for i, v in enumerate(vecs):
            score = cosine_similarity(query_vector, v)
            res.append((score, i))
        res.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Carregamento de Metadados (Lazy Load)
        if self.s3_client:
            resp = self.s3_client.get_object(Bucket=self.s3_bucket, Key=meta_path)
            meta_data = json.loads(resp['Body'].read().decode('utf-8'))
        else:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        
        global_offset = sum(c_idx[i][1] for i in range(best_id))
        
        final_output = []
        for score, local_idx in res[:top_k]:
            g_idx = global_offset + local_idx
            m = meta_data.get(str(g_idx))
            if m:
                content = m["content"]
                # Expansão de Contexto
                if margin_chars > 0:
                    p = meta_data.get(str(g_idx-1), {}).get("content", "")[-margin_chars:] if g_idx > 0 else ""
                    s = meta_data.get(str(g_idx+1), {}).get("content", "")[:margin_chars]
                    content = f"{p}{content}{s}"
                
                final_output.append({
                    "score": score,
                    "content": content,
                    "metadata": m["metadata"]
                })
        return final_output
