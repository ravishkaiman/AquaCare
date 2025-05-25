import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


class FishClassifier(nn.Module):
    def __init__(self, num_classes=31):
        super(FishClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 31
num_epochs = 100
batch_size = 32
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder("train", transform=transform)
val_dataset = datasets.ImageFolder("train", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = FishClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")


    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    train_accuracy = 100.0 * correct / total
    print(f"Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")


    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)


            progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    val_accuracy = 100.0 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "fish_classifier_model.pth")
print("Training complete. Model saved.")
