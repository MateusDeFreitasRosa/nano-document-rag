import os
import random
from nano_rag import NanoRAG

# --- SIMULAÇÃO: PROCESSO OFFLINE (Ingestão) ---
def offline_ingestion_example():
    print("--- [OFFLINE] Iniciando Ingestão de Dados ---")
    
    # 1. Seus dados (Textos e Metadados)
    chunks = [
        "O NanoRAG é uma biblioteca Python focada em eficiência para AWS Lambda.",
        "A busca vetorial utiliza K-Means para agrupar vetores e acelerar a busca.",
        "O armazenamento binário .vlog permite leitura parcial de dados do disco.",
        "Você pode usar qualquer modelo de embedding, como OpenAI ou Cohere.",
        "A grama é verde e o céu é azul em dias ensolarados."
    ]
    
    metadatas = [
        {"source": "docs", "page": 1},
        {"source": "docs", "page": 2},
        {"source": "tech_specs", "page": 1},
        {"source": "api_ref", "page": 5},
        {"source": "natureza", "page": 10}
    ]

    # 2. Simulação de Geração de Embeddings (Vetor de dimensão 3 para o exemplo)
    # Na vida real, você usaria: client.embeddings.create(...) da OpenAI
    dim = 3
    embeddings = [[random.uniform(-1, 1) for _ in range(dim)] for _ in chunks]

    # 3. Indexação com NanoRAG
    rag = NanoRAG(db_name="meu_banco")
    
    # Usando o método facilitador 'fit'
    rag.fit(
        embeddings=embeddings,
        contents=chunks,
        metadatas=metadatas
    )
    print("--- [OFFLINE] Indexação Concluída com Sucesso! ---\n")


# --- SIMULAÇÃO: PROCESSO ONLINE (Busca na Lambda) ---
def online_query_example():
    print("--- [ONLINE] Iniciando Busca Vetorial ---")
    
    # 1. Carrega o NanoRAG (Apenas leitura do índice existente)
    rag = NanoRAG(db_name="meu_banco")
    
    # 2. Simulação de um Vetor de Query (gerado pelo mesmo modelo do offline)
    # Vamos criar um vetor aleatório de dimensão 3
    query_vector = [random.uniform(-1, 1) for _ in range(3)]
    
    # 3. Realiza a busca
    results = rag.query(query_vector, top_k=2)
    
    print(f"Resultados para a query {query_vector}:")
    for i, res in enumerate(results):
        print(f"\nResultado #{i+1}:")
        print(f"  Score: {res['score']:.4f}")
        print(f"  Conteúdo: {res['content']}")
        print(f"  Metadata: {res['metadata']}")
        print(f"  Cluster ID: {res['cluster_id']}")
    
    print("\n--- [ONLINE] Busca Finalizada ---")


if __name__ == "__main__":
    # Executa o fluxo completo
    offline_ingestion_example()
    online_query_example()

    # Limpeza opcional dos arquivos gerados no exemplo
    # os.remove("meu_index.vlog")
    # os.remove("meu_metadata.json")
