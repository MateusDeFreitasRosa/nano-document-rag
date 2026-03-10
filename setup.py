from setuptools import setup, find_packages

setup(
    name="nano-document-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[], 
    author="Mateus de Freitas Rosa",
    description="Uma biblioteca ultra-lightweight para RAG focada em documentos e otimizada para AWS Lambda",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MateusDeFreitasRosa/nano-rag",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
