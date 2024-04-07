from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from baseline import *
import numpy as np
import torch.optim as optim

# model
model = CNNBaseline()

# other external model settings
num_epochs = 10
batch_size = 16
train_acc, train_loss, val_acc, val_loss = [], [], [], []

# data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder('./data', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# setting metrics
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training the model
for epoch in range(num_epochs):
    num_correct = 0
    epoch_loss = 0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predictions = outputs.max(1)
        num_correct += (predictions == labels).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_acc.append(num_correct / batch_size)
    train_loss.append(epoch_loss / batch_size)
    
    
        
    print('Train: Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
