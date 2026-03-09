from setuptools import setup, find_packages

setup(
    name="nano-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[], # Zero dependências!
    author="Seu Nome",
    description="Uma biblioteca lightweight para RAG otimizada para Serverless (AWS Lambda)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/nano-rag",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
