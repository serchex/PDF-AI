from huggingface_hub import InferenceClient
import pdf
from collections import defaultdict

PDF_PATH = "./pdfs/hechos.pdf"

pages = pdf.build_pages(PDF_PATH)

client = InferenceClient(
    token="TU_TOKEN"
)

def extract_local_context(text, name, window=200):
    t = text.lower()
    n = name.lower()

    idx = t.find(n)
    if idx == -1:
        return text

    start = max(0, idx - window)
    end = min(len(text), idx + window)

    return text[start:end]

def question(pregunta, context):
    prompt = f"""
        Usa solo el texto siguiente.

        Texto:
        {context}

        Pregunta:
        ¿{pregunta}?


        Responde de forma clara y breve usando solo información explícita del texto.
    """

    resp = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "Responde solo con información explícita del texto. No inventes nada."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    return resp.choices[0].message["content"]

def infer_role(name, context):
    prompt = f"""
        Usa solo el texto siguiente.

        Texto:
        {context}

        Instrucción:
        Extrae literalmente la frase más cercana al nombre "{name}" que describa
        su cargo, función, quien es o título profesional.


        Responde usando exactamente las palabras del texto.

        - Si no hay ninguna frase que describa un cargo, quien es la persona o función, responde exactamente:
        No determinado
    """

    resp = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "Responde solo con información explícita del texto. No inventes nada."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    return resp.choices[0].message["content"]

raw_input = input("Busca nombres (separados por coma): ")
pregunta = input('> ')

names = [n.strip().lower() for n in raw_input.split(",")]
print("NAMES:", names)
#buscar menciones
all_hits = {}
for name in names:
    hits = pdf.find_mentions(pages, name, threshold=60)
    grouped = pdf.group_by_page(hits)
    contexts = pdf.build_context(grouped)
    #print('contexts:', contexts)
    all_hits[name] = contexts
    print(f"\nBuscando: {name}")
    print("Hits encontrados:", len(hits))
results = defaultdict(list)
#print('results: ',results)

for name, contexts in all_hits.items():
    if not contexts:
        print(f"No se encontró contexto para {name}")
    for ctx in contexts:
        local_ctx = extract_local_context(ctx["context"], name)
        role = infer_role(name, local_ctx)
        #print(role)
        # role = infer_role(name, ctx["context"])
        results[name].append({
            "page": ctx["page"],
            "role": role,
            "evidence": ctx["context"][:]
        })
        print(question(pregunta, ctx["context"]))

for name, items in results.items():
    print(f"\n=== {name.upper()} ===")
    for it in items:
        print(f"- Página {it['page']}: {it['role']}")
    if not results:
        print("No se pudo inferir ningún rol. Resultados vacíos.")


