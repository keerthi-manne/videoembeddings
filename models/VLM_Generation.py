import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
class VLM_Generation() : 
    # Load model (use base version for efficiency)
    def __init__(self, config_path: str = 'configs/model_base.json'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()
    def generate_caption(self, image, prompt=None):
        """
        image: can be numpy array (OpenCV) OR PIL image
        """
        if isinstance(image, Image.Image):
            raw_image = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            raw_image = Image.fromarray(image_rgb)

        # Process input
        if prompt:
            inputs = self.processor(raw_image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        out = self.model.generate(
            **inputs,
            max_length=20,
            num_beams=3
        )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

# if __name__ == "__main__":
#     VLM = VLM_Generation()
#     image_path = "test.jpg"

#     # Example classifier output
#     label = "tree"
#     confidence = 0.3

#     if confidence > 0.4:
#         prompt = f"{label}"
#     else:
#         prompt = None

#     caption = VLM.generate_caption(image_path, prompt)

#     print("Generated Caption:", caption)