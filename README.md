# NanoRAG

Uma biblioteca Python lightweight para RAG (Retrieval-Augmented Generation) otimizada para ambientes Serverless (AWS Lambda). A lib foca em busca vetorial eficiente ($O(\log N)$) utilizando Clustering (K-Means) e armazenamento binário, sem dependências pesadas como FAISS, NumPy ou Pandas.

**Nota:** Esta biblioteca não gera embeddings. Você deve fornecer os vetores já gerados (ex: via OpenAI, Cohere, HuggingFace).

## Características

*   **Zero Heavy Libs:** Apenas Python standard library (`math`, `struct`, `json`, `random`).
*   **Serverless Friendly:** Otimizada para baixo consumo de memória e inicialização rápida.
*   **Busca em 2 Etapas:** Usa centróides para filtrar o espaço de busca, carregando apenas o cluster relevante do disco.
*   **Armazenamento Binário:** Vetores salvos em formato binário customizado (`.vlog`) para I/O rápido.

## Instalação

```bash
pip install nano-rag
```
*(Ou copie a pasta `nano_rag` para o seu projeto)*

## Como Usar

### 1. Inserção de Dados (Indexação)

Você tem duas formas profissionais de indexar seus dados:

#### Opção A: Método Facilitador (`fit`)
Ideal quando você tem listas separadas de vetores e textos.

```python
from nano_rag import NanoRAG

embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
contents = ["Texto do chunk 1", "Texto do chunk 2"]
metadatas = [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}] # Opcional

rag = NanoRAG()
rag.fit(embeddings, contents, metadatas)
```

#### Opção B: Método de Documentos (`insert`)
Ideal quando você já tem os objetos estruturados.

```python
documents = [
    {
        "content": "O céu é azul.",
        "embedding": [0.1, 0.2, ...],
        "metadata": {"source": "natureza"}
    }
]

rag.insert(documents)
```

### 2. Busca (Query)

Para buscar, você deve fornecer o **vetor** da sua query. A biblioteca não converte texto em vetor.

```python
# Vetor da query (gerado pelo mesmo modelo usado na indexação)
query_vector = [0.15, 0.25, 0.75, ...] 

results = rag.query(query_vector, top_k=3)

for res in results:
    print(f"Score: {res['score']:.4f}")
    print(f"Conteúdo: {res['content']}")
    print(f"Metadata: {res['metadata']}")
    print("-" * 20)
```

## Estrutura de Arquivos

*   `index.vlog`: Arquivo binário contendo os vetores e centróides.
*   `metadata.json`: Arquivo JSON mapeando IDs de vetores para o conteúdo textual.
