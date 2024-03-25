# imports for embeddings
import requests
from retry import retry
import pandas as pd
        
# Embedding our chunks {transfrom chunks of PDFs to Embeddings}
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

# The first time you generate the embeddings it may take a while (approximately 20 seconds) for the API to return them. We use the retry decorator (install with pip install retry) so 
# that if on the first try output = query(dict(inputs = texts)) doesn't work, wait 10 seconds and try again three times. The reason this happens is because on the first request, the model needs to be downloaded and installed in the server, but subsequent calls are much faster.
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

@retry(tries=3, delay=10)
def Embedding_func(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
       raise RuntimeError(
          "The model is currently loading, please re-run the query."
       )
    
texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans’ Benefits?"]

output = Embedding_func(texts)

embeddings = pd.DataFrame(output)
# print(embeddings)