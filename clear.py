from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np
import torch
from PIL import ImageEnhance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from PIL import Image, ImageFilter

img = Image.open("docsr_output.png").convert("RGB")

denoised = img.filter(ImageFilter.MedianFilter(size=1))
denoised.save("step1_denoised.png")

denoise = img.filter(ImageFilter.GaussianBlur(radius=0.08))
denoise.save("denoise_gaussian.png")

gray = denoise.convert("L")
contrast1 = ImageEnhance.Contrast(gray).enhance(1.2)
contrast2 = ImageEnhance.Contrast(contrast1).enhance(1.2)

contrast2.save("contrast_double.png")

sharp = contrast2.filter(ImageFilter.UnsharpMask(
    radius=2,
    percent=120,
    threshold=3
))
sharp.save("sharp_unsharp.png")

gray = sharp.convert("L")

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",
    model=model,
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device=device
)

img_np = np.array(gray)
output, _ = upsampler.enhance(img_np, outscale=2)
sr = Image.fromarray(output)
sr.save("step2_sr.png")

contrast = ImageEnhance.Contrast(sr).enhance(0.1)
sharp = ImageEnhance.Sharpness(contrast).enhance(0.1)

sharp.save("step3_edges.png")
