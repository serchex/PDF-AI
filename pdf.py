import fitz
from PIL import Image
from pytesseract import image_to_string
import io
import re

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
                'content': re.sub(r'\n{2,}','\n',texto)
            })
        if img:
            ocr_txt = img_tobytes(img,page).strip()
            if ocr_txt:
                bloques.append({
                    'page': i+1,
                    'source': 'ocr',
                    'content': re.sub(r'\n{2,}','\n',ocr_txt)
                })
    return bloques
