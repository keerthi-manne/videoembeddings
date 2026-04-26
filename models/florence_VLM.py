import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import cv2
import sys
import types
import importlib.machinery

flash_attn_mock = types.ModuleType("flash_attn")
flash_attn_mock.__spec__ = importlib.machinery.ModuleSpec(
    name="flash_attn",
    loader=None
)

sys.modules["flash_attn"] = flash_attn_mock
class VLM_Generation():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load Florence-2
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=self.torch_dtype, trust_remote_code=True  ).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)


    def generate_batch(self, images, prompts):
        pil_images = []

        for img in images:
            if not isinstance(img, Image.Image):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            pil_images.append(img)

        inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3
            )

        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )

        results = []
        for text, img, prompt in zip(generated_texts, pil_images, prompts):
            parsed = self.processor.post_process_generation(
                text,
                task=prompt,
                image_size=(img.width, img.height)
            )
            result_txt = parsed[prompt] 
            clean_text = result_txt.replace("<pad>", "").strip()
            results.append(clean_text)
        return results