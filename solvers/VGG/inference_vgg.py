import numpy as np
from argparse import ArgumentParser
from torchvision import models, transforms, datasets
import torch
import torch.nn as nn
import pandas as pd
import csv
from torch.autograd import Variable


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Преобразование изображений для подготовки к классификации
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main(args):
    model = models.vgg16(pretrained=False)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 5)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()


    data_dir = args.data_dir
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for inputs, labels in dataloader:
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        print(predicted)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", help='directory with pngs', required=True)
    parser.add_argument("--weights_path", help='path to weights', required=True)

    args = parser.parse_args()

    main(args)
