import fitz
from PIL import Image
from pytesseract import image_to_string
import io
import re
from rapidfuzz import fuzz
import cv2
import clear
import numpy as np

def clean_text(text: str) -> str:
    text = text.replace('\r', '\n')
    #caracteres basura OCR
    text = re.sub(r'[|~¬]', '', text)
    #unir letras separadas: E J E M P L O
    text = re.sub(r'(?<=\b[A-Z])\s+(?=[A-Z]\b)', '', text)
    #reducir espacios multiples
    text = re.sub(r' {2,}', ' ', text)
    #reducir saltos de linea excesivos
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def texto_puro(page):
    texto = page.get_text()
    return texto

def OCR(img):
    image = Image.open(img)
    texto = image_to_string(image, lang='spa')
    return texto

def img_incrustada(page,doc):
    for img_index, img in enumerate(page.get_images()):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n < 5:
            pix.save(f'Imagen_extraida_{img_index}.png')
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(f'imagen_extraida_{img_index}.png')

#carga la imagen a la RAM
def img_tobytes(img,page):
    pix = page.get_pixmap(dpi=img[0][3])
    img_bytes = pix.tobytes("png")
    imagen = Image.open(io.BytesIO(img_bytes))
    texto = image_to_string(imagen)
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
                'content': clean_text(texto)
            })
        if img:
            ocr_txt = img_tobytes(img,page).strip()
            if ocr_txt:
                bloques.append({
                    'page': i+1,
                    'source': 'ocr',
                    'content': clean_text(ocr_txt)
                })
    return bloques

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
