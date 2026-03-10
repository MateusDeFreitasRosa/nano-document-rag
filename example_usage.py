import os
import random
from nano_rag import NanoDocumentRAG

# --- CONFIGURAÇÕES DO EXEMPLO ---
STORAGE_DIR = "./meu_storage"
DOC_ID = "manual_tecnico_v1"

# --- SIMULAÇÃO: PROCESSO OFFLINE (Geração do Índice) ---
def exemplo_indexacao_offline():
    """
    Simula o momento em que você processa seus documentos localmente
    ou em um job de CI/CD para gerar os arquivos .vlog e .json.
    """
    print("\n🚀 [OFFLINE] Iniciando Indexação do Documento...")
    
    # 1. Seus dados (Textos e Metadados)
    chunks = [
        "O NanoDocumentRAG é focado em eficiência para AWS Lambda.",
        "A busca vetorial utiliza K-Means para agrupar vetores e acelerar a busca.",
        "O armazenamento binário .vlog permite leitura parcial (Range Requests).",
        "A expansão de contexto dinâmica ajuda o LLM a entender melhor o assunto.",
        "O céu é azul e a grama é verde em dias de sol intenso."
    ]
    
    # 2. Simulação de Embeddings (Vetor de dimensão 3 para o exemplo)
    # Na vida real: embeddings = openai_client.embeddings.create(input=chunks, ...)
    dim = 3
    embeddings = [[random.uniform(-1, 1) for _ in range(dim)] for _ in chunks]

    # 3. Criando o índice físico (Gera os arquivos .vlog e .json)
    rag = NanoDocumentRAG(document_id=DOC_ID, storage_dir=STORAGE_DIR)
    rag.index(embeddings, chunks)
    
    print(f"✅ [OFFLINE] Índice criado com sucesso em: {STORAGE_DIR}")


# --- SIMULAÇÃO: PROCESSO ONLINE (Busca no AWS Lambda) ---
def exemplo_busca_online_local():
    """
    Simula o uso da biblioteca dentro de uma AWS Lambda usando 
    armazenamento local (ex: EFS montado ou arquivos no /tmp).
    """
    print("\n🔍 [ONLINE - LOCAL] Buscando com Expansão de Contexto...")
    
    # Inicializa o motor apontando para a pasta onde os índices estão
    rag = NanoDocumentRAG(document_id=DOC_ID, storage_dir=STORAGE_DIR)
    
    # Vetor da query (gerado pelo mesmo modelo da indexação)
    query_vector = [random.uniform(-1, 1) for _ in range(3)]
    
    # Busca com 50 caracteres de margem (pega trechos dos vizinhos)
    results = rag.search(query_vector, top_k=1, margin_chars=50)
    
    for res in results:
        print(f"🎯 Match (Score: {res['score']:.4f})")
        print(f"📝 Conteúdo Expandido: \"{res['content']}\"")


def exemplo_busca_online_s3():
    """
    Exemplo real de como usar a biblioteca dentro de uma AWS Lambda buscando
    diretamente do Amazon S3 sem baixar o arquivo inteiro (Range Requests).
    """
    print("\n☁️ [ONLINE - S3] Buscando diretamente do S3...")
    
    # 1. Em uma Lambda real, o boto3 já está disponível:
    # import boto3
    # s3 = boto3.client('s3')
    
    # Simulando o objeto s3_client para demonstração
    class MockS3Client:
        def get_object(self, Bucket, Key, Range=None):
            # Simula o comportamento do boto3.client('s3').get_object()
            print(f"   [S3 API CALL] get_object(Bucket='{Bucket}', Key='{Key}', Range='{Range}')")
            # Em um cenário real, isso retornaria o streaming de bytes do S3
            raise NotImplementedError("Este é apenas um exemplo conceitual. Use um s3_client real do boto3.")

    s3_mock = MockS3Client()
    
    print("💡 Configurando o NanoDocumentRAG para modo S3:")
    
    # Inicializa o motor injetando o cliente S3 e as coordenadas do bucket
    # A lib detecta o s3_client e entra automaticamente no modo 'Range Request'
    rag = NanoDocumentRAG(
        document_id=DOC_ID,
        storage_dir="indices",      # Pasta/Prefixo dentro do bucket
        s3_client=s3_mock,          # Injeção do cliente boto3
        s3_bucket="meu-bucket-rag"  # Nome do seu bucket S3
    )
    
    print(f"   rag = NanoDocumentRAG(document_id='{DOC_ID}', s3_client=s3, s3_bucket='meu-bucket-rag')")
    
    try:
        # A busca agora faria chamadas parciais ao S3 via rede
        # results = rag.search(query_vector, margin_chars=200)
        print("   ✅ Pronto para realizar buscas ultra-eficientes no S3!")
    except Exception as e:
        print(f"   ℹ️ {e}")


if __name__ == "__main__":
    # 1. Primeiro geramos os arquivos localmente
    exemplo_indexacao_offline()
    
    # 2. Depois testamos a busca local (como se fosse um EFS ou /tmp)
    exemplo_busca_online_local()
    
    # 3. Demonstração conceitual do S3
    exemplo_busca_online_s3()
    
    print("\n✨ Exemplo finalizado!")
