from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import render
import os
import ollama
import base64

def chat_screen(request):
    return render(request, "chat.html")

def sendMessage(request):
    if request.method == "POST":
        message = request.POST.get("message", "")
        uploaded_file = request.FILES.get("file")

        file_url = None
        file_name = None

        if uploaded_file:
            file_name = uploaded_file.name
            file_path = os.path.join("uploads", file_name)
            saved_path = default_storage.save(file_path, ContentFile(uploaded_file.read()))
            file_url = default_storage.url(saved_path)
            bot_response = infer_image(saved_path, prompt=message)
        else:
            bot_response = infer(message)

        return JsonResponse({"response": bot_response, "file_url": file_url, "file_name": file_name})

    return JsonResponse({"error": "Invalid request"}, status=400)
    
def infer(prompt):

    response = ollama.chat(
        model="llama3.2-vision",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    
    return response["message"]["content"]

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