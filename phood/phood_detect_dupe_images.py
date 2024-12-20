import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load images
def load_images(image_folder):
    images = []
    file_names = []
    for file in os.listdir(image_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(image_folder, file)
            images.append(Image.open(file_path).convert('RGB'))
            file_names.append(file_path)
    return images, file_names

# Compute image hashes (or embeddings for similarity)
def compute_image_embeddings(images):
    embeddings = []
    for image in images:
        resized_image = image.resize((128, 128))  # Standardize size
        embeddings.append(np.array(resized_image).flatten())
    return np.array(embeddings)

# Find duplicates
def find_duplicates(embeddings, threshold=0.95):
    similarities = cosine_similarity(embeddings)
    duplicates = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            if similarities[i, j] > threshold:
                duplicates.append((i, j))
    return duplicates

image_folder = "path_to_image_folder"
images, file_names = load_images(image_folder)
embeddings = compute_image_embeddings(images)
duplicates = find_duplicates(embeddings)

# Remove duplicates
for i, j in duplicates:
    os.remove(file_names[j])  # Remove duplicate