# 🚀 NanoDocumentRAG

Uma biblioteca Python **ultra-lightweight** para RAG (Retrieval-Augmented Generation) focada em documentos individuais e otimizada para ambientes **Serverless (AWS Lambda)**.

O **NanoDocumentRAG** utiliza uma estratégia de busca por centróides ($O(\log N)$) e armazenamento binário customizado (`.vlog`) para garantir performance máxima com consumo mínimo de memória.

---

## 💡 Por que usar o NanoDocumentRAG?

*   **Zero Heavy Libs:** Sem NumPy, Pandas ou FAISS. Apenas Python standard library.
*   **Foco em Documentos:** Cada documento gera seu próprio índice físico isolado.
*   **Serverless Friendly:** Cold start zero e baixíssimo uso de RAM.
*   **Contextual Expansion:** Expansão dinâmica de contexto (janela de vizinhos) na hora da busca.
*   **Cálculo Automático:** Otimização automática de clusters ($K = \sqrt{N}$).

---

## 📦 Instalação

```bash
pip install git+https://github.com/MateusDeFreitasRosa/nano-rag.git
```

---

## 🛠️ Como Usar

### 1. Indexação (Processo Offline)
Nesta etapa, você gera o índice vetorial para um documento específico. Você deve fornecer os embeddings e os textos já processados.

```python
from nano_rag import NanoDocumentRAG

# Configurações
doc_id = "manual_tecnico_v1"
storage = "./meu_storage"

# Dados (Ex: vindos de uma API de Embedding)
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
chunks = ["O NanoDocumentRAG é eficiente.", "Ideal para AWS Lambda."]
metadatas = [{"page": 1}, {"page": 2}] # Opcional

# Inicializa e indexa
rag = NanoDocumentRAG(document_id=doc_id, storage_dir=storage)
rag.index(embeddings, chunks, metadatas)
# Isso gera: ./meu_storage/manual_tecnico_v1.vlog e .json
```

### 2. Busca com Expansão de Contexto (Processo Online)
Na hora de buscar, você pode definir uma margem de caracteres para incluir o contexto dos chunks vizinhos.

```python
from nano_rag import NanoDocumentRAG

# Carrega o motor para o documento
rag = NanoDocumentRAG(document_id="manual_tecnico_v1", storage_dir="./meu_storage")

# Vetor da query (gerado pelo mesmo modelo da indexação)
query_vector = [0.15, 0.25, ...]

# Busca com expansão de 200 caracteres para cada lado (vizinhos)
results = rag.search(query_vector, top_k=3, margin_chars=200)

for res in results:
    print(f"Score: {res['score']:.4f}")
    print(f"Conteúdo (Expandido): {res['content']}")
    print(f"Metadata: {res['metadata']}")
    print("-" * 20)
```

---

## 📂 Estrutura de Armazenamento

Para cada documento indexado, o NanoDocumentRAG cria dois arquivos:

1.  **`{document_id}.vlog`**: Arquivo binário (`float32`) contendo os vetores e centróides. Otimizado para leitura parcial via *disk seek*.
2.  **`{document_id}.json`**: Arquivo de metadados contendo os textos originais e informações extras.

---

## ⚙️ Requisitos
*   Python 3.7+
*   Zero dependências externas.
