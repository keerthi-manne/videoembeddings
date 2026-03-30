import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


class VLM_Generation():
    def __init__(self):
        # 🔥 Load model (ViT + GPT2 captioning)
        self.model_name = "ydshieh/vit-gpt2-coco-en"

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_single(self, image):
        # 🔹 Convert image to PIL if needed
        if not isinstance(image, Image.Image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # 🔥 Preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # 🔥 Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=16,
                num_beams=1
            )

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return caption.strip()

    def generate_batch(self, images):
        pil_images = []

        for img in images:
            if not isinstance(img, Image.Image):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            pil_images.append(img)

        pixel_values = self.processor(images=pil_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=16,
                num_beams=1
            )

        captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions = [c.strip() for c in captions]

        return captions