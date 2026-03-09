# NanoRAG

Uma biblioteca Python lightweight para RAG (Retrieval-Augmented Generation) otimizada para ambientes Serverless (AWS Lambda). A lib foca em busca vetorial eficiente ($O(\log N)$) dentro de **documentos individuais** utilizando Clustering (K-Means) e armazenamento binário, sem dependências pesadas.

**Nota:** Esta biblioteca não gera embeddings. Você deve fornecer os vetores já gerados (ex: via OpenAI, Cohere, HuggingFace).

## Características

*   **Zero Heavy Libs:** Apenas Python standard library (`math`, `struct`, `json`, `random`).
*   **Foco em Documentos:** Cada documento possui seu próprio índice físico isolado.
*   **Serverless Friendly:** Otimizada para baixo consumo de memória e inicialização rápida.
*   **Busca em 2 Etapas:** Usa centróides para filtrar o espaço de busca dentro do documento.

## Instalação

```bash
pip install nano-rag
```

## Como Usar

### 1. Indexação de um Documento

Você deve fornecer os vetores (embeddings) e os textos (chunks) de um documento específico.

```python
from nano_rag import NanoRAG

# Dados do documento
doc_id = "manual_tecnico_v1"
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
contents = ["Texto do chunk 1", "Texto do chunk 2"]
metadatas = [{"page": 1}, {"page": 2}] # Opcional

# Inicializa o motor para este documento
rag = NanoRAG(document_id=doc_id, storage_dir="./indices")

# Gera o índice (cria ./indices/manual_tecnico_v1.vlog e .json)
rag.index(embeddings, contents, metadatas)
```

### 2. Busca em um Documento (Search)

Para buscar, você deve carregar o motor com o `document_id` correspondente.

```python
# Carrega o motor para o documento desejado
rag = NanoRAG(document_id="manual_tecnico_v1", storage_dir="./indices")

# Vetor da query (gerado pelo mesmo modelo usado na indexação)
query_vector = [0.15, 0.25, 0.75, ...] 

# Busca os trechos mais similares dentro deste documento
results = rag.search(query_vector, top_k=3)

for res in results:
    print(f"Score: {res['score']:.4f}")
    print(f"Conteúdo: {res['content']}")
    print(f"Metadata: {res['metadata']}")
    print("-" * 20)
```

## Estrutura de Arquivos

*   `{document_id}.vlog`: Arquivo binário contendo os vetores e centróides do documento.
*   `{document_id}.json`: Arquivo JSON mapeando os vetores para o conteúdo textual.
