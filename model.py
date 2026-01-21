import torch
from sentence_transformers import SentenceTransformer
import chromadb
import textwrap
from huggingface_hub import InferenceClient
import pdf
import uuid

chroma = chromadb.Client()
try:
    collection = chroma.create_collection("my_docs")
except:
    collection = chroma.get_collection("my_docs")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using: ',device)

bloques = pdf.pdf_text('zorro_merged.pdf')

def chunk_text(texto, size=400):
    return textwrap.wrap(texto,width=size)

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
for b in bloques:
    chunks = chunk_text(b['content'], size=1000)
    for chunk in chunks:
        vec = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[vec],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                'page': b['page'],
                'source': b['source']
            }]
        )

pregunta = 'De que color eran los ojos del zorro'
while pregunta != 'salir':
    pregunta = input()
    query_emb = model.encode([pregunta]).tolist()
    resultado = collection.query(
        query_embeddings=query_emb, 
        n_results=3,
        #where={'source':'txt'},
        include=['documents','embeddings','metadatas','distances'],
        )
    print(resultado['distances'], resultado['metadatas'])

    # print(resultado['distances'][0])
    # print(max(resultado['distances'][0]))
    if resultado['distances'][0][0] > 1:
        resultado = collection.query(
            query_embeddings=query_emb,
            n_results=3,
            where={'source':'text'},
            #include=['documents','embeddings','metadatas','distances']
        )
    else:
        resultado = collection.query(
            query_embeddings=query_emb,
            n_results=3,
            where={'source':'ocr'},
            #include=['documents','embeddings','metadatas','distances']
        )
        #print('else:',resultado['distances'], resultado['metadatas'])
    print(resultado['distances'], resultado['metadatas'])


    client = InferenceClient(
        #'Qwen/Qwen2.5-1.5B-Instruct',
        #token = 'YOUR HUGGING FACE TOKEN' #unmark this line
    )

    contexto = " ".join(
        f"[Pagina {m['page']} | {m['source']}]\n{d}"
        for sub_docs, sub_meta in zip(resultado['documents'], resultado['metadatas'])
        for d, m in zip(sub_docs,sub_meta)
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

    print("\n\nRESPUESTA:")
    print(resp.choices[0].message['content'])
