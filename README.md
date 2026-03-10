# 🚀 NanoDocumentRAG

Uma biblioteca Python **ultra-lightweight** para RAG (Retrieval-Augmented Generation) focada em **documentos individuais** e otimizada para ambientes **Serverless (AWS Lambda)**.

O objetivo do **NanoDocumentRAG** é oferecer um motor de busca vetorial que não dependa de infraestruturas pesadas. Ele permite que você gerencie seus documentos como arquivos isolados, armazenando-os localmente ou diretamente no **Amazon S3**, realizando buscas ultra-eficientes sem carregar o banco inteiro na memória.

---

## 🎯 Por que NanoDocumentRAG?

*   **Zero Heavy Libs:** Sem NumPy, Pandas ou FAISS. Apenas Python standard library.
*   **Foco em Documentos:** Cada documento é um índice físico isolado (`.vlog` + `.json`).
*   **S3 Native:** Busca no S3 usando *Range Requests* (baixa apenas os bytes necessários).
*   **Smart Append:** Adicione novos dados a documentos existentes com re-clusterização automática.
*   **Contextual Expansion:** Expanda o contexto da busca dinamicamente (pega trechos dos vizinhos).
*   **Serverless Ready:** Cold start zero, ideal para AWS Lambda e instâncias pequenas.

---

## 📦 Instalação

```bash
# Apenas o Core (Local/EFS)
pip install git+https://github.com/MateusDeFreitasRosa/nano-rag.git

# Com suporte a S3 (AWS)
pip install "nano-document-rag[aws] @ git+https://github.com/MateusDeFreitasRosa/nano-rag.git"
```

---

## 🛠️ Como Usar: Caso de Uso Completo

### 1. Preparação e Indexação (Processo Offline)
Ideal para rodar em um SageMaker, GitHub Action ou máquina local.

```python
from nano_rag import NanoDocumentRAG, smart_chunk_text

# 1. Configura o storage (pode ser local ou S3)
rag = NanoDocumentRAG(
    storage_dir="meus_indices",
    s3_bucket="meu-bucket-rag" # Opcional: ativa upload automático
)

# 2. Prepara o texto (Smart Chunking preserva palavras completas)
texto = "O NanoDocumentRAG é eficiente..."
chunks = smart_chunk_text(texto, max_chars=500, overlap=100)

# 3. Gera embeddings (use sua API de preferência)
# embeddings = openai_client.embeddings.create(input=chunks, ...)
embeddings = [[0.1, 0.2, ...], ...] 

# 4. Cria o documento (ou adiciona dados a um existente)
rag.create_or_replace_document("manual_tecnico_v1", embeddings, chunks)
# rag.add_to_document("manual_tecnico_v1", novos_embeddings, novos_chunks)
```

### 2. Busca de Alta Performance (Processo Online)
Ideal para rodar em AWS Lambda. A lib busca apenas o necessário via rede.

```python
import boto3
from nano_rag import NanoDocumentRAG

# Injeta o cliente S3 (já disponível no Lambda)
s3 = boto3.client('s3')

rag = NanoDocumentRAG(
    storage_dir="meus_indices",
    s3_client=s3,
    s3_bucket="meu-bucket-rag"
)

# Busca em um ou mais documentos simultaneamente
query_vector = [0.15, 0.25, ...]
results = rag.search(
    document_id=["manual_tecnico_v1", "contrato_v2"],
    query_vector=query_vector,
    top_k=3,
    margin_chars=200 # Expande o contexto com 200 caracteres dos vizinhos
)

for res in results:
    print(f"Documento: {res['document_id']} | Score: {res['score']:.4f}")
    print(f"Conteúdo: {res['content']}")
```

---

## 📂 Estrutura de Armazenamento

*   **`{doc_id}.vlog`**: Binário (`float32`) com vetores e centróides. Suporta *disk seek*.
*   **`{doc_id}.json`**: Metadados e textos originais.

## ⚙️ Requisitos
*   Python 3.7+
*   `boto3` (opcional, apenas para uso com S3)
