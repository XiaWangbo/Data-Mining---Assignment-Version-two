import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cnn_model import CNNModel
import torch.optim as optim
import torch.nn as nn
import json

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    # Training
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}, Training loss: {train_loss:.4f}")

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Epoch {epoch}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

# Save the trained model
model_save_path = "trained_model.pt"
torch.save(model.state_dict(), model_save_path)

# Save the accuracy
accuracy_save_path = "model_accuracy.json"
with open(accuracy_save_path, 'w') as f:
    json.dump({"accuracy": accuracy}, f)
