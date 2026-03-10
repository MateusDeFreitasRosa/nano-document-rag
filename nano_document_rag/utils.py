def smart_chunk_text(text, max_chars=1000, overlap=200):
    """
    Divide um texto em chunks respeitando os limites de palavras completas.
    
    Args:
        text: O texto original a ser dividido.
        max_chars: Tamanho máximo aproximado de cada chunk.
        overlap: Quantidade de caracteres de sobreposição entre chunks.
        
    Returns:
        List[str]: Lista de chunks com palavras preservadas.
    """
    if not text:
        return []
        
    chunks = []
    text_len = len(text)
    start = 0

    while start < text_len:
        # Define o fim teórico
        end = start + max_chars
        
        # Se não chegamos no fim do texto, ajustamos para não cortar palavra
        if end < text_len:
            # Procura o próximo espaço, quebra de linha ou pontuação
            # Tentamos encontrar o fim da palavra atual
            next_space = text.find(" ", end)
            next_newline = text.find("\n", end)
            
            # Pega o separador mais próximo
            candidates = [c for c in [next_space, next_newline] if c != -1]
            
            if candidates:
                actual_end = min(candidates)
                # Só expandimos se o "pulo" não for absurdamente grande (limite de 100 chars)
                if (actual_end - end) < 100:
                    end = actual_end
        
        # Extrai o chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calcula o próximo início considerando o overlap
        start = end - overlap
        
        # Ajusta o início para não começar no meio de uma palavra (anda para o próximo espaço)
        if start > 0 and start < text_len:
            next_start_space = text.find(" ", start)
            if next_start_space != -1 and next_start_space < end:
                start = next_start_space + 1
        
        # Garante que o loop avance
        if start <= chunks[-1].find(chunk) + len(chunks[-1]) if chunks else 0:
            if start < end:
                start = end
            else:
                start += 1

    return chunks
