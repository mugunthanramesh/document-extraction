import torch
from PIL import Image
import open_clip
import os

import ollama
import base64

def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def infer_image(image_path, prompt="Describe this image"):
    """Runs inference on the image using the local Ollama model."""
    image_base64 = encode_image(image_path)
    
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[
            {"role": "user", "content": prompt, "images": [image_base64]},
        ]
    )
    
    return response["message"]["content"]


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

resource_dir = "/Users/mugunthan/github/document-extraction/resources"

images = []
image_paths = []
questions = ["where is the elephant present"]


for document in os.listdir(resource_dir):
    if document.endswith(("png", "jpg", "jpeg")):
        image_path = Image.open(os.path.join(resource_dir, document)).convert("RGB")
        image = preprocess(image_path).unsqueeze(0)
        images.append(image)
        image_paths.append(os.path.join(resource_dir, document))

question = input("Enter the question: ")
image_tensor = torch.cat(images).to('cpu')
text = tokenizer([question]).to('cpu')

with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)

    best_match_index = similarity.argmax().item()
    best_match = image_paths[best_match_index]

    print(f"best match for \"{question}\" is {best_match}")
    answer = infer_image(best_match, question)
    print(f"Answer: \n\n{answer}")
    print("---------------------------------------------------")
