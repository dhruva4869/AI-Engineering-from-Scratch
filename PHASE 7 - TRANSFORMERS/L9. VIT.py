"""
2020 se pehle computer vision ka king CNNs (Convolutional Neural Networks) the. ImageNet, object detection, sab jagah CNN backbones use hote the. 
Transformers mostly NLP ke liye use hote the.
Phir 2020 mein paper An Image is Worth 16x16 Words ne idea diya: image ko chhote 16x16 patches mein tod do, 
har patch ko ek token ki tarah treat karo, aur unhe normal Transformer encoder mein bhej do. Isse Vision Transformer (ViT) bana.

Simple idea:
Sentence = words → tokens
Image = patches → tokens

Image -> patches (patchify) -> Tokens Vectors -> Simple Conv2d -> Positional Encodings -> Transformer -> (Encoder & Decoder later for image generation)

Why it took a while
ViT needs a lot of data to match CNNs because it has none of the CNN inductive biases (translation invariance, locality). Without >100M labeled images or strong self-supervised pretraining, 
CNNs still win at matched compute. DeiT fixed this in 2021 with distillation tricks; DINOv2 fixed it permanently in 2023 with self-supervision.
Deit -> cnn ko use krke badhaya
DINO -> khud se learn krke improved
"""

def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches


from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation

# ek image representation will be combination of many such vectors

