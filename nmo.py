from huggingface_hub import InferenceClient
import pdf
from collections import defaultdict

PDF_PATH = "step2_sr.pdf"

pages = pdf.build_pages(PDF_PATH)

client = InferenceClient(
    token="TOKEN_HUGGINGFACE"
)

def infer_role(name, context):
    prompt = f"""
        Usa solo el texto siguiente.

        Texto:
        {context}

        Pregunta:
        ¿Qué rol juega la persona llamada "{name}" en este texto?

        Responde con una frase corta.
        Si el rol no es explícito, responde exactamente:
        "No determinado"
    """

    resp = client.chat_completion(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "Responde solo con información explícita del texto. No inventes nada."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    return resp.choices[0].message["content"]

while 1:
    raw_input = input("Busca nombres (separados por coma): ")
    names = [n.strip().lower() for n in raw_input.split(",")]

    #buscar menciones
    all_hits = {}

    for name in names:
        hits = pdf.find_mentions(pages, name, threshold=80)
        grouped = pdf.group_by_page(hits)
        contexts = pdf.build_context(grouped)
        all_hits[name] = contexts




    results = defaultdict(list)

    for name, contexts in all_hits.items():
        for ctx in contexts:
            role = infer_role(name, ctx["context"])
            results[name].append({
                "page": ctx["page"],
                "role": role,
                "evidence": ctx["context"][:300]
            })

    for name, items in results.items():
        print(f"\n=== {name.upper()} ===")
        for it in items:
            print(f"- Página {it['page']}: {it['role']}")