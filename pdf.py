import fitz
from PIL import Image
import io
from rapidfuzz import fuzz
import numpy as np
Image.MAX_IMAGE_PIXELS = None
import ocr_img

# def texto_puro(page):
#     texto = page.get_text()
#     return texto

# def OCR(img):
#     image = Image.open(img)
#     texto = image_to_string(image, lang='spa')
#     return texto

# def img_incrustada(page,doc):
#     for img_index, img in enumerate(page.get_images()):
#         xref = img[0]
#         pix = fitz.Pixmap(doc, xref)
#         if pix.n < 5:
#             pix.save(f'Imagen_extraida_{img_index}.png')
#         else:
#             pix = fitz.Pixmap(fitz.csRGB, pix)
#             pix.save(f'imagen_extraida_{img_index}.png')

#carga la imagen a la RAM
def img_tobytes(img,page):
    dpi = min(img[0][3], 300)
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    imagen = Image.open(io.BytesIO(img_bytes))

    MAX_SIZE = (2000, 2000)
    imagen.thumbnail(MAX_SIZE, Image.LANCZOS)
    np_img = np.array(imagen)
    texto = ocr_img.ocr_clean(np_img)
    return texto


def pdf_text(opt):
    doc = fitz.open(opt)
    bloques = []
    for i, page in enumerate(doc):
        texto = page.get_text()
        img = page.get_images()
        if texto:
            bloques.append({
                'page': i+1,
                'source': 'text',
                'content': ocr_img.clean_text(texto)
            })
        if img:
            ocr_txt = img_tobytes(img,page).strip()
            if ocr_txt:
                bloques.append({
                    'page': i+1,
                    'source': 'ocr',
                    'content': ocr_txt
                })
    return bloques

print(pdf_text('2.pdf'))

def build_pages(pdf_path):
    bloques = pdf_text(pdf_path)
    pages = []

    for b in bloques:
        pages.append({
            "doc_id": pdf_path,
            "page": b["page"],
            "source": b["source"],
            "text": b["content"]
        })

    return pages


def find_mentions(pages, name, threshold=80):
    name = name.lower()
    hits = []

    for p in pages:
        text = p["text"].lower()
        score = fuzz.partial_ratio(name, text)

        if score >= threshold:
            hits.append({
                "doc_id": p["doc_id"],
                "page": p["page"],
                "source": p["source"],
                "score": score,
                "text": p["text"]
            })

    return hits

from collections import defaultdict

def group_by_page(hits):
    grouped = defaultdict(list)
    for h in hits:
        key = (h["doc_id"], h["page"])
        grouped[key].append(h)
    return grouped

def build_context(grouped_hits):
    contexts = []

    for (doc_id, page), items in grouped_hits.items():
        text = items[0]["text"]
        contexts.append({
            "doc_id": doc_id,
            "page": page,
            "context": text
        })

    return contexts

# def remove_noise_lines(text: str) -> str:
#     lines = text.splitlines()
#     clean = []
#     for line in lines:
#         letters = re.findall(r'[A-Za-zÁÉÍÓÚÑáéíóúñ]', line)
#         if len(letters) >= 5:
#             clean.append(line)
#     return '\n'.join(clean)
# def fix_ocr_confusions(text: str) -> str:
#     text = re.sub(r'(?<=[A-Za-zÁÉÍÓÚÑáéíóúñ])0(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])', 'O', text)
#     text = re.sub(r'(?<=\d)[lI](?=\d)', '1', text)
#     return text
# def clean_text(text: str) -> str:
#     text = unicodedata.normalize("NFKC", text)
#     text = text.replace('\r', '\n')
#     text = re.sub(r'[|~¬<>:=\[\]\(\)\{\}]', ' ', text)
#     text = re.sub(r'(?<=\b[A-ZÁÉÍÓÚÑ])\s+(?=[A-ZÁÉÍÓÚÑ]\b)', '', text)
#     text = fix_ocr_confusions(text)
#     text = remove_noise_lines(text)
#     text = re.sub(r'\n(?=[a-záéíóúñ])', ' ', text, flags=re.IGNORECASE)
#     text = re.sub(r'\s{2,}', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()
