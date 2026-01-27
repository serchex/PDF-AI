import torch
import numpy as np
from PIL import Image
from docsr import DocSR

#config
IMAGE_PATH = "images/carta_poder.png"
OUTPUT_PATH = "docsr_output.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#cargar imagen en grayscale
img = Image.open(IMAGE_PATH).convert("L")

#upscale primero
img = img.resize(
    (img.width * 2, img.height * 2),
    Image.BICUBIC
)

img_np = np.array(img).astype(np.float32) / 255.0
img_np = img_np[np.newaxis, np.newaxis, :, :] #NCHW

x = torch.from_numpy(img_np).to(DEVICE)

#cargar modelo
model = DocSR().to(DEVICE)
model.eval()

#modelo sin entrenar = baseline

with torch.no_grad():
    y = model(x)

out = y.squeeze().cpu().numpy()
out = np.clip(out * 255, 0, 255).astype(np.uint8)

Image.fromarray(out).save(OUTPUT_PATH)
print("DocSR output guardado:", OUTPUT_PATH)
