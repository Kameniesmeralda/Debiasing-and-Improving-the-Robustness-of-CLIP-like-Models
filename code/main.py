import torch
from PIL import Image
import open_clip
from urllib.request import urlopen

# Load the model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load an image from the web (a golden retriever)
url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_100.jpg"
image = preprocess(Image.open(urlopen(url))).unsqueeze(0)

# Define labels
labels = ["a diagram", "a dog", "a cat"]
text = tokenizer(labels)

# Run Inference
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate probabilities
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Print results
print(f"Image source: {url}")
print("Label probabilities:")
for label, prob in zip(labels, text_probs[0]):
    print(f"{label}: {prob:.4f}")