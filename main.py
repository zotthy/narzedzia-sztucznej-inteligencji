import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


train_data = pd.read_csv('/Users/sebastianstarzec/siecineur/mnist_train.csv')
test_data = pd.read_csv('/Users/sebastianstarzec/siecineur/mnist_test.csv')

custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor(self.data.iloc[idx, 0], dtype=torch.long)
        image = torch.tensor(self.data.iloc[idx, 1:].values, dtype=torch.float32).view(
            1, 28, 28) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = CustomDataset(train_data, transform=custom_transform)
test_dataset = CustomDataset(test_data, transform=custom_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Definicja funkcji straty i optymizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Trenowanie modelu
epochs = 3000 
start_time = time.time()
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Ewaluacja modelu na zbiorze testowym
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy}')

print('Training complete!')
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training complete! Time elapsed: {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.2f} seconds')

model.eval() 

correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on test data: {accuracy * 100:.2f}%')


# zapis modelu

model = NeuralNetwork()
torch.save(model.state_dict, 'model.pth')
model.eval()


num_examples_to_display = 10
examples_to_display = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for j in range(len(images)):
            examples_to_display.append((images[j].cpu(), labels[j].cpu(), predicted[j].cpu()))

        if len(examples_to_display) >= num_examples_to_display:
            break

plt.figure(figsize=(15, 10))

num_examples_to_display = len(examples_to_display)

num_rows = (num_examples_to_display // 4)
num_cols = min(num_examples_to_display, 4)

for i, (image, true_label, predicted_label) in enumerate(examples_to_display):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"True class: {true_label}, Predicted class: {predicted_label}")
    plt.axis('off')
    plt.axis('off')

plt.tight_layout()
plt.savefig('wynik.png')
plt.show()