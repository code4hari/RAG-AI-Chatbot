import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

links_df = pd.read_csv('document_links.csv')
document_links = dict(zip(links_df['file_name'], links_df['link']))

folder_path = 'docs'
pdf_text = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.pdf'):
        reader = PdfReader(os.path.join(folder_path, file_name))
        link = document_links.get(file_name, 'No link available')
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            text += f" ({file_name}, page {i + 1}, link: {link})"
            pdf_text.append({'index': len(pdf_text), 'content': text})

df = pd.DataFrame(pdf_text)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True)

df['embeddings'] = embeddings.tolist()

import pinecone
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="quickstart",
    dimension=len(embeddings[0]),
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
index = pc.Index("quickstart")

to_upsert = [
    {
        "id": str(row['index']),
        "values": row['embeddings'],
        "metadata": {"content": row['content']}
    }
    for _, row in df.iterrows()
]
index.upsert(vectors=to_upsert)
