import torch
from sentence_transformers import SentenceTransformer
import chromadb
import textwrap
from huggingface_hub import InferenceClient
import pdf
import uuid
from sklearn.preprocessing import normalize
import hdbscan
from collections import Counter,defaultdict
import numpy as np

chroma = chromadb.Client()
try:
    collection = chroma.create_collection("my_docs")
except:
    collection = chroma.get_collection("my_docs")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using: ',device)

bloques = pdf.pdf_text('zorro_merged.pdf')

# def chunk_text(texto, size=400):
#     return textwrap.wrap(texto,width=size)
def chunk_text(text, size=400, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

all_chunks = []
all_embeddings = []
all_metadatas = []


for b in bloques:
    chunks = chunk_text(b['content'], size=700)
    for chunk in chunks:
        if len(chunk.split()) < 30:
            continue
        vec = model.encode(chunk)
        all_chunks.append(chunk)
        all_embeddings.append(vec)
        all_metadatas.append({
            'page': b['page'],
            'source': b['source']
        })

embeddings_norm = normalize(all_embeddings)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric='euclidean',
    cluster_selection_method='leaf'
)

labels = clusterer.fit_predict(embeddings_norm)

for chunk, vec, meta, label in zip(all_chunks,all_embeddings,all_metadatas, labels):
    meta['cluster_id'] = int(label)
    
    collection.add(
        documents=[chunk],
        embeddings=[vec.tolist()],
        ids=[str(uuid.uuid4())],
        metadatas=[meta]
    )
print(labels)
print(Counter(labels))
print(clusterer.condensed_tree_.to_pandas().head())



clusters_text = defaultdict(list)
for chunk, meta in zip(all_chunks, all_metadatas):
    cid = meta['cluster_id']
    clusters_text[cid].append({
        'text': chunk,
        'page': meta['page'],
        'source': meta['source']
    })

for cid, items in clusters_text.items():
    print("\n" + "="*80)
    print(f"CLUSTER {cid}  (n={len(items)})")
    print("="*80)

    for i, item in enumerate(items):
        print(f"\n--- Chunk {i+1} | Page {item['page']} | {item['source']} ---")
        print(item['text'][:500])  # solo los primeros 500 chars
################
pregunta = 'De que color eran los ojos del zorro'
while pregunta != 'salir':
    clusters = defaultdict(list)

    pregunta = input()
    query_emb = model.encode([pregunta]).tolist()
    resultado = collection.query(
        query_embeddings=query_emb, 
        n_results=3,
        #where={'source':'txt'},
        include=['documents','embeddings','metadatas','distances'],
        )
    print(resultado['distances'], resultado['metadatas'])
    ###
    docs = resultado['documents'][0]
    metas = resultado['metadatas'][0]
    dists = resultado['distances'][0]
    for doc, meta, dist in zip(docs, metas, dists):
        cid = meta['cluster_id']
        #if cid != -1:
        clusters[cid].append(dist)

    best_cluster = min(
        clusters.items(),
        key=lambda x: np.mean(x[1])
    )[0]

    print(best_cluster)
    ###
    client = InferenceClient(
        #'Qwen/Qwen2.5-1.5B-Instruct',
        token = 'TOKEN_HUGGINGFACE',        
    )

    contexto = " ".join(
        f"[Pagina {m['page']} | {m['source']}]\n{d}"

        for d, m in zip(docs, metas)
        if m['cluster_id'] == best_cluster
    )

    prompt = f"""
    Usa solo la siguiente información para responder.
    Información:
    {contexto}

    Pregunta: ¿{pregunta}?
    """

    resp = client.chat_completion(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "Responde solo con información textual exacta del contexto. No inventes nada. Si la respuesta no está explícita en el texto, responde: ""No está en el texto."" No describas, no infieras, no completes, no adornes. No agregues adjetivos que no aparezcan en el texto. Si la intruccion es ""citar texto"" o intrucciones similares, responde con una frase breve que este dentro del texto original, no inventes y acompletes ni cambies nada del texto"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    # resp = client.text_generation(
    #     model="Qwen/Qwen2.5-1.5B-Instruct",
    #     prompt=prompt,
    #     max_new_tokens=200
    # )
    # print("\n\nRESPUESTA:")
    # print(resp)
    print(resp.choices[0].message['content'])
