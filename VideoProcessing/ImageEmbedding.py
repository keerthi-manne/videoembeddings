import torch
import clip
from PIL import Image
# Load model
def process_image(image, device , model , preprocess ):

    # Generate embedding
    with torch.no_grad():
        image_embedding = model.encode_image(image)

    # Normalize (important!)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding
