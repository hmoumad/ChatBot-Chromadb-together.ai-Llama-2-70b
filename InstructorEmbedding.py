# Requirements:
# After you start, you may have a problem "No module named transformers, Dependencies for InstructorEmbedding not found":

    # to solve it Reinstalling the libraries and make sure that t=you already ahve all of them.
        # pip install huggingface
        # pip install transformers
        # pip install InstructorEmbedding
        # pip install torch
        # pip install sentence_transformers

from langchain_community.embeddings import HuggingFaceEmbeddings

def embed_query(text):

    embeddings = HuggingFaceEmbeddings()
    query_result = embeddings.embed_query(text)
    return query_result 


# Example usage
text = "This is a test document."
result = embed_query(text)
print(result[:3])
