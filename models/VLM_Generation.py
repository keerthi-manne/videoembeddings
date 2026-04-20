import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2

class VLM_Generation():
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.processor.tokenizer.padding_side = "left"
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)



    def generate_batch(self, images , prompts ):
        pil_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            pil_images.append(img)
        inputs = self.processor(images=pil_images, text = prompts  , return_tensors="pt" , padding = True , truncation = True ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2
            )

        # 🔥 Decode captions
        captions = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        captions = [c.strip() for c in captions]

        return captions