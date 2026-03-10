import math

def dot_product(v1, v2):
    """Calculates the dot product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    """Calculates the magnitude (Euclidean norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    m1 = magnitude(v1)
    m2 = magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot_product(v1, v2) / (m1 * m2)

def euclidean_distance(v1, v2):
    """Calculates the Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

def vector_add(v1, v2):
    """Adds two vectors element-wise."""
    return [x + y for x, y in zip(v1, v2)]

def vector_scale(v, scalar):
    """Scales a vector by a scalar."""
    return [x * scalar for x in v]
