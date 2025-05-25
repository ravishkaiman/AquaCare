import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # For real-time progress bars

# ============================
# 1. Configuration
# ============================
DATA_DIR = "disease_dataset"  # Update with your dataset path
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2  # 20% for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. Data Transforms
# ============================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# 3. Datasets and DataLoaders
# ============================
def get_dataloaders():
    # Load the full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * VALID_SPLIT)
    train_size = dataset_size - val_size

    # Split dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Apply validation transforms to disable data augmentation for the validation dataset
    val_dataset.dataset.transform = val_transforms

    # Print the class names and number of classes
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    print("Class names:", class_names)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, num_classes

# ============================
# 4. Model Setup
# ============================
def initialize_model(num_classes):
    # Use a pretrained ResNet50
    model = models.resnet50(pretrained=True)
    # Optionally freeze base layers by uncommenting the loop below:
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)
    return model

# ============================
# 5. Training and Validation Functions
# ============================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap the DataLoader with tqdm for a progress bar
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update the progress bar with current loss and accuracy
        progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ============================
# 6. Main Training Loop
# ============================
def main(num_epochs):
    train_loader, val_loader, num_classes = get_dataloaders()
    model = initialize_model(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        # Validate after the training epoch
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fish_disease_model.pth")

        # Print epoch loss and accuracy
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%")
        print("-" * 50)

    print("Training complete. Best validation accuracy: {:.2f}%".format(best_val_acc * 100))

if __name__ == '__main__':
    # For Windows, especially if freezing to an executable, use freeze_support()
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except ImportError:
        pass

    # Parse command line arguments to allow custom settings for number of epochs
    parser = argparse.ArgumentParser(description='Train a fish disease detection model.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    # Use the custom number of epochs from command line input
    main(num_epochs=args.epochs)
