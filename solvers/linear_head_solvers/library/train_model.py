import torch
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shutil
from tqdm import tqdm
import pandas as pd
from loguru import logger
import cv2

import os
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger
from math import sqrt
from scipy import stats
import os
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np




class FolderDataset(Dataset):

    def __init__(self, img_paths, img_labels,  processor):
        self.paths = img_paths
        self.labels = img_labels
        self.processor = processor


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.paths[index]
        label = self.labels[index]
        img = Image.open(path).convert("RGB")
        img = self.processor(img)
        return img, label, path



def calculate_metrics(all_labels, all_preds, test_type="train"):
    
    all_probs = np.array(all_probs).astype(np.float64)
    all_probs = np.array(all_labels).astype(np.float64)
    
    test_srcc, _ = stats.spearmanr(all_preds, all_labels)
    test_plcc, _ = stats.pearsonr(all_preds, all_labels)
    return {f'{test_type}_Test_SRCC': test_srcc, f'{test_type}_Test_PLCC': test_plcc})


class ModelTrainer:
    def __init__(self, train_df, val_df, model, criterion, optimizer, scheduler, num_epochs, n_epoch_val,
                 model_name, model_predict, processor, data_folder, batch_size, num_workers):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        train_imgs, train_labels = train_df.img_path.tolist(), train_df.final_mos.tolist()
        val_imgs, val_labels = val_df.img_path.tolist(), train_df.final_mos.tolist()
        
        self.train_ds = FolderDataset(train_imgs, train_labels, processor=processor)
        self.val_ds = FolderDataset(val_imgs, val_labels, processor=processor)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model_name = model_name

        
        self.model_predict = model_predict
    

    def run(self):
        since = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)
            self.model, train_metrics, epoch_loss = self.train_model(epoch)
            self.model, val_metrics = self.val_model_pytorch(epoch)
            epoch_results = {
            torch.save(self.model.state_dict(), f"{self.weights_folder}/model_{epoch}.pt")


    def train_model(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds, all_labels = [], []
        all_probs, all_paths = [], []
        
        for inputs, labels, paths in tqdm(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            probs = self.model_predict(inputs, grad=True, model=self.model, device=self.device)
            loss = self.criterion(probs, labels.float())
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * len(labels)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)
            
        self.scheduler.step()
        
        epoch_loss = running_loss / len(self.train_ds)
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        metrics = calculate_metrics(all_labels, all_probs)
        return self.model, metrics, epoch_loss
                    

    def val_model_pytorch(self):
        self.model.eval()
        all_preds, all_labels = [], []
        all_probs, all_paths = [], []
        infer_begin = time.time()
        batch_infer_time = []
        
        for inputs, labels, paths in tqdm(self.val_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            probs = self.model_predict(inputs, grad=False, model=self.model, device=self.device)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        metrics_data = calculate_metrics(all_labels, all_preds)
        return self.model, metrics_data
