import torch
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

    def __len__(self):
        return len(self.paths)



def calculate_metrics(all_labels, all_preds, test_type="train"):
    
    all_probs = np.array(all_preds).astype(np.float64)
    all_probs = np.array(all_labels).astype(np.float64)
    
    test_srcc, _ = stats.spearmanr(all_preds, all_labels)
    test_plcc, _ = stats.pearsonr(all_preds, all_labels)
    return {f'{test_type}_SRCC': test_srcc, f'{test_type}_PLCC': test_plcc}


class ModelTrainer:
    def __init__(self, df, model, criterion, optimizer, scheduler, num_epochs, n_epoch_val,
                 model_name, model_predict, processor, data_folder, batch_size, num_workers):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        train_imgs, train_labels = train_df.img_path.tolist(), train_df.final_mos.tolist()
        val_imgs, val_labels = val_df.img_path.tolist(), train_df.final_mos.tolist()
        
        self.train_ds = FolderDataset(train_imgs, train_labels, processor=processor)
        self.val_ds = FolderDataset(val_imgs, val_labels, processor=processor)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.model_name = model_name
        self.results = []
        self.model_predict = model_predict

        self.save_every_k_epoch = 10

        self.model_folder = os.path.join(data_folder, "experiments", model_name)
        
        self.metrics_path = os.path.join(self.model_folder, f"metrics.csv")
        self.weights_folder =  os.path.join(self.model_folder, "weights")
        os.makedirs(self.weights_folder, exist_ok=True)

    

    def run(self):
        since = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)
            self.model, train_metrics, epoch_loss = self.train_model(epoch)
            self.model, val_metrics = self.val_model_pytorch()
            epoch_results = {"epoch": epoch, "train_epoch_loss": epoch_loss, **train_metrics, **val_metrics}
            print(epoch_results)
            self.results.append(epoch_results)
            pd.DataFrame(self.results).to_csv(self.metrics_path, index=False)
            if epoch % self.save_every_k_epoch == 0:
                torch.save(self.model.state_dict(), f"{self.weights_folder}/model_{epoch}.pt")


    def train_model(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds, all_labels = [], []
        all_probs, all_paths = [], []
        
        for inputs, labels, paths in tqdm(self.train_loader, desc="Train model"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            probs = self.model_predict(inputs, grad=True, model=self.model, device=self.device)
            loss = self.criterion(probs, labels.float())
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * len(labels)
                        
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)
            
        self.scheduler.step()
        
        epoch_loss = running_loss / len(self.train_ds)
        
        metrics = calculate_metrics(all_labels, all_probs, "train")
        return self.model, metrics, epoch_loss
                    

    def val_model_pytorch(self):
        self.model.eval()
        all_preds, all_labels = [], []
        all_probs, all_paths = [], []
        infer_begin = time.time()
        batch_infer_time = []
        
        for inputs, labels, paths in tqdm(self.val_loader, desc="Validate model"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            probs = self.model_predict(inputs, grad=False, model=self.model, device=self.device)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_paths.extend(paths)

        metrics_data = calculate_metrics(all_labels, all_probs, "val")
        return self.model, metrics_data
