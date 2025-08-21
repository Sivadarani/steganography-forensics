import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. Device setup (CPU only)
device = torch.device("cpu")
print(f"Using device: {device}")

# 2. Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder("dataset/train", transform=transform)
val_data = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

# 3. Load ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Clean, Stego
model = model.to(device)  # move model to CPU

# 4. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(5):  # small demo
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # move data to CPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 6. Save model
torch.save(model.state_dict(), "models/best_stego_resnet18.pth")
print("✅ Model saved as models/best_stego_resnet18.pth")
