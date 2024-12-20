import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# Define SimCLR Augmentations
class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter()], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def __call__(self, x):
        return self.transform(x)

# Load dataset
dataset = ImageFolder(root="path_to_data", transform=SimCLRTransform())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define SimCLR Model
class SimCLR(nn.Module):
    def __init__(self, base_model):
        super(SimCLR, self).__init__()
        self.backbone = base_model
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return z

base_model = models.resnet50(pretrained=True)
base_model.fc = nn.Identity()  # Remove final classification layer
model = SimCLR(base_model)

# Training Loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for (x, _) in dataloader:
        x1, x2 = x  # Two augmented views
        z1 = model(x1)
        z2 = model(x2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()