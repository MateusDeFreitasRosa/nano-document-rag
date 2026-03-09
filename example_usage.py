import os
import random
from nano_rag import NanoRAG

# --- SIMULAÇÃO: PROCESSO OFFLINE (Indexação de Documentos) ---
def offline_indexing_example():
    print("--- [OFFLINE] Indexando Documentos Individuais ---")
    
    # 1. Dados do Documento A
    doc_a_id = "manual_tecnico"
    chunks_a = [
        "O NanoRAG é focado em eficiência para AWS Lambda.",
        "A busca vetorial utiliza K-Means para agrupar vetores.",
        "O armazenamento binário .vlog permite leitura parcial."
    ]
    # Simulação de embeddings (dimensão 3)
    embeddings_a = [[random.uniform(-1, 1) for _ in range(3)] for _ in chunks_a]

    # 2. Dados do Documento B
    doc_b_id = "contrato_venda"
    chunks_b = [
        "Este contrato rege a venda de serviços de software.",
        "O prazo de entrega é de 30 dias úteis.",
        "A multa por rescisão antecipada é de 10%."
    ]
    embeddings_b = [[random.uniform(-1, 1) for _ in range(3)] for _ in chunks_b]

    # 3. Criando os índices (em uma pasta específica)
    storage = "./meu_storage"
    
    # Indexando Documento A
    rag_a = NanoRAG(document_id=doc_a_id, storage_dir=storage)
    rag_a.index(embeddings_a, chunks_a)
    
    # Indexando Documento B
    rag_b = NanoRAG(document_id=doc_b_id, storage_dir=storage)
    rag_b.index(embeddings_b, chunks_b)
    
    print(f"--- [OFFLINE] Índices criados em: {storage} ---\n")


# --- SIMULAÇÃO: PROCESSO ONLINE (Busca Direcionada) ---
def online_search_example():
    print("--- [ONLINE] Buscando em um Documento Específico ---")
    
    storage = "./meu_storage"
    
    # O usuário quer buscar apenas no 'manual_tecnico'
    doc_id = "manual_tecnico"
    rag = NanoRAG(document_id=doc_id, storage_dir=storage)
    
    # Vetor da query (dimensão 3)
    query_vector = [random.uniform(-1, 1) for _ in range(3)]
    
    # Busca
    results = rag.search(query_vector, top_k=2)
    
    print(f"Resultados encontrados no documento '{doc_id}':")
    for res in results:
        print(f"  Score: {res['score']:.4f} | Texto: {res['content']}")
    
    print("\n--- [ONLINE] Busca Finalizada ---")


if __name__ == "__main__":
    offline_indexing_example()
    online_search_example()
