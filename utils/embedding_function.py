from langchain_community.embeddings.ollama import OllamaEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function(name:str="multilingual-e5-base"):
    if name == "mxbai-embed-large":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large",
                                  base_url='http://host.docker.internal:11434')
    elif name == "nomic-embed-text":
        embeddings = OllamaEmbeddings(model="nomic-embed-text",
                                  base_url='http://host.docker.internal:11434')
    elif name == "multilingual-e5-base":
        hugging_embed = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        embeddings = create_langchain_embedding(hugging_embed)

    return embeddings