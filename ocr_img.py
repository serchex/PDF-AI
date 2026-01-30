import pytesseract
import cv2
from PIL import Image
import numpy as np
import unicodedata
import re
import clear

def remove_noise_lines(text: str) -> str:
    lines = text.splitlines()
    clean = []

    for line in lines:
        letters = re.findall(r'[A-Za-zÁÉÍÓÚÑáéíóúñ]', line)
        if len(letters) >= 5: #ajusta para ruido
            clean.append(line)

    return '\n'.join(clean)


def fix_ocr_confusions(text: str) -> str:
    #0 entre letras -> O
    text = re.sub(r'(?<=[A-Za-zÁÉÍÓÚÑáéíóúñ])0(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])', 'O', text)

    #l o I entre números -> 1
    text = re.sub(r'(?<=\d)[lI](?=\d)', '1', text)

    return text


def clean_text(text: str) -> str:
    #normalizacion  Unicode (acentos, sibolos raros)
    text = unicodedata.normalize("NFKC", text)

    #normalizar saltos de linea
    text = text.replace('\r', '\n')

    #quitar caracteres basura OCR comunes
    text = re.sub(r'[|~¬<>:=\[\]\(\)\{\}]', ' ', text)

    #unir letras separadas: E J E M P L O
    text = re.sub(r'(?<=\b[A-ZÁÉÍÓÚÑ])\s+(?=[A-ZÁÉÍÓÚÑ]\b)', '', text)

    #corregir confusiones OCR (0/O, l/1)
    text = fix_ocr_confusions(text)

    #eliminar líneas basura
    text = remove_noise_lines(text)

    #unir líneas cortadas por OCR
    text = re.sub(r'\n(?=[a-záéíóúñ])', ' ', text, flags=re.IGNORECASE)

    #reducir espacios múltiples
    text = re.sub(r'\s{2,}', ' ', text)

    #reducir saltos de línea excesivos
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)

def detect_orientation(image):
    osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    angle = osd.get("rotate", 0)
    return angle

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def ocr_clean(image_np_rgb):
    """
    image_np_rgb: np.ndarray (H, W, 3) en RGB
    return: texto limpio
    """

    # OpenCV trabaja en BGR
    edges = clear.ganear(image_np_rgb, True)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Deskew
    im = deskew(edges_bgr)

    #orientacion
    angle = detect_orientation(im)
    rotated = rotate_image(im, angle)
    cv2.imwrite("step_rotated.png", rotated)
    #gris
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    #OCR
    text = pytesseract.image_to_string(
        gray,
        lang="spa",
        config="--oem 3 --psm 6"
    )

    clean = clean_text(text)

    return clean


##FUncion para pruebas!!!!
def image_to_numpy(img):
    """
    Recibe:
    - PIL.Image.Image
    - ruta a imagen (str)
    - np.ndarray

    Devuelve:
    - np.ndarray en RGB (H, W, 3)
    """

    if isinstance(img, np.ndarray):
        return img

    if isinstance(img, Image.Image):
        return np.array(img)

    if isinstance(img, str):
        img_pil = Image.open(img).convert("RGB")
        return np.array(img_pil)

    raise TypeError("Tipo de imagen no soportado")
#print(ocr_clean(image_to_numpy('./images/2.jpeg')))
#guardar a archivo
# with open("ocr_output.txt", "w", encoding="utf-8") as f:
#     f.write(text)
