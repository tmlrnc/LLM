from transformers import AutoModelForImageClassification, AutoProcessor
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]

# Load pre-trained model
model_name = "google/vit-base-patch16-224-in21k"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Predict labels
dataset = CustomImageDataset(file_names, transform=processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

predicted_labels = []
model.eval()
with torch.no_grad():
    for images, paths in dataloader:
        inputs = processor(images, return_tensors="pt")
        outputs = model(**inputs)
        predicted_labels.extend(outputs.logits.argmax(dim=-1).tolist())

# Save results for review
results = list(zip(file_names, predicted_labels))