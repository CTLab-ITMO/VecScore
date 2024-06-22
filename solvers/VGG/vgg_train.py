import os
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, models, transforms
import torch.nn as nn
from torchvision.io import read_image
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm

# Определение аугментаций для обучающего и валидационного наборов данных
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path[:-4] + '.png').convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])
        if label > 5:
            label = 5
        if label < 1:
            label = 1
        # print(label)
        label_vec = torch.zeros(5)
        label_vec[label - 1] = 1.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label_vec

# Пример использования
annotations_file = 'good_data.csv'
img_dir = 'png-pictures'
transform = ToTensor()

dataset = CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform=val_transform)
subset_size = 10

# Создайте индексы для подмножества
indices = torch.randperm(len(dataset)).tolist()
subset_indices = indices[:subset_size]

# Создайте DataLoader с SubsetRandomSampler
sampler = SubsetRandomSampler(subset_indices)
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)


val_size = 0.2
num_train = len(dataset)
num_val = int(num_train * val_size)
num_train = num_train - num_val

# Разделите датасет
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

# Создайте DataLoader'ы для обучающего и валидационного наборов
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Загрузка предобученной модели VGG16
model = models.vgg16(pretrained=True)

num_classes = 5
# Замена последнего полносвязного слоя
# Заморозьте все слои, кроме последнего классификационного слоя

# Замените последний классификационный слой на новый, подходящий для вашего набора данных
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features, num_classes)  # num_classes - количество ваших классов

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print('TRAIN_START')
# Функция для обучения модели
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        for param in model.features.parameters():
            param.requires_grad = False
        running_loss = 0.0
        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}')

        # Валидация модели
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, target = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == target).sum().item()
        print(f'Validation Loss: {val_loss/len(val_dataloader)}, Accuracy: {100 * correct / total}%')

# Обучение модели
train_model(model, criterion, optimizer, num_epochs=5)
