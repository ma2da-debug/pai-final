"""CLIP(ゼロショット画像・テキスト照合)で、言葉に合う物体を選ぶ層。"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"


class Matcher:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def rank(self, query, crops):
        """各切り抜きと query の類似度(候補間のsoftmax)を返す。"""
        images = [Image.fromarray(c).resize((224, 224), Image.BICUBIC) for c in crops]
        text = f"a photo of {query}, seen from above"
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        logits = self.model(**inputs).logits_per_text[0]
        return torch.softmax(logits, 0).numpy()
