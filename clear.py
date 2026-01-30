from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def ganear(img_np_rgb, debug=False):
    """
    img_np_rgb: np.ndarray (H, W, 3) RGB
    return: np.ndarray (H, W) GRAYSCALE  para OCR
    """

    #Denoise
    img = Image.fromarray(img_np_rgb)
    denoised = img.filter(ImageFilter.MedianFilter(size=1))

    #Grayscale/ normalización
    gray = denoised.convert("L")
    gray_np = np.array(gray).astype(np.float32)
    gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())
    gray_np = (gray_np * 255).astype(np.uint8)

    gray_rgb = Image.fromarray(gray_np).convert("RGB")
    img_np = np.array(gray_rgb)

    #super-resolution
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

    output, _ = upsampler.enhance(img_np, outscale=2)

    if debug:
        Image.fromarray(output).save("step3_sr.png")

    #Binarizacion
    sr_gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        sr_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        15
    )

    if debug:
        cv2.imwrite("step4_binary.png", binary)

    #Refinado de bordes
    kernel = np.array([
        [0, -1, 0],
        [-1, 6, -1], #ajusta entre 5 y 8
        [0, -1, 0]
    ])
    edges = cv2.filter2D(binary, -1, kernel)

    if debug:
        cv2.imwrite("step5_edges_refined.png", edges)

    return edges


