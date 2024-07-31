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
import shutil


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


def log_images(image_paths: list[str], titles, plot_path):
    image_paths, titles = image_paths[:50], titles[:50]
    num_rows = int(sqrt(len(image_paths)))
    num_cols = int(sqrt(len(image_paths)))
    print(num_rows, num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(13*num_cols, 10*num_rows))
    images = [cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (512, 512)) for img_path in image_paths]
    for i, img in enumerate(images[:(num_rows*num_cols)]):
        # img = plt.imshow(img)
        row = i // num_cols
        col = i % num_cols
        # print(row, col)
        ax = axes[row][col]
        if titles:
            ax.set_title(str(titles[i]))
        ax.imshow(img)
        ax.axis('off')
    plt.savefig(plot_path)


def calculate_metrics(all_labels, all_preds, img_paths, test_type="train"):
    
    all_probs_ar = np.array(all_preds).astype(np.float64)
    all_labels_ar = np.array(all_labels).astype(np.float64)
    
    test_srcc, _ = stats.spearmanr(all_probs_ar, all_labels_ar)
    test_plcc, _ = stats.pearsonr(all_probs_ar, all_labels_ar)
    
    # Calculate absolute error
    abs_error = np.abs(all_labels_ar - all_probs_ar)
    
    # Calculate absolute error
    abs_error = np.abs(all_labels_ar - all_probs_ar)
    
    # Get indices of 20 worst predictions
    worst_indices = np.argsort(abs_error)[-20:]
    
    # Create a list of dictionaries for the 20 worst predictions
    worst_images_info = []
    for idx in worst_indices:
        worst_images_info.append({
            'pred_score': all_probs_ar[idx],
            'label': all_labels_ar[idx],
            'img_path': img_paths[idx]
        })
    
    return {f'{test_type}_SRCC': test_srcc, f'{test_type}_PLCC': test_plcc, f'worst_images': worst_images_info}


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

        self.save_every_k_epoch = 2

        self.model_folder = os.path.join(data_folder, "experiments", model_name)
        shutil.rmtree(self.model_folder, ignore_errors=True)
        
        self.metrics_path = os.path.join(self.model_folder, f"metrics.csv")
        self.weights_folder =  os.path.join(self.model_folder, "weights")
        os.makedirs(self.weights_folder, exist_ok=True)

    

    # def run(self):
    #     since = time.time()
        
    #     for epoch in range(1, self.num_epochs + 1):
    #         print(f'Epoch {epoch}/{self.num_epochs - 1}')
    #         print('-' * 10)
    #         self.model, train_metrics, epoch_loss = self.train_model(epoch)
    #         self.model, val_metrics = self.val_model_pytorch()
    #         epoch_results = {"epoch": epoch, "train_epoch_loss": epoch_loss, **train_metrics, **val_metrics}
    #         print(epoch_results)
    #         self.results.append(epoch_results)
    #         pd.DataFrame(self.results).to_csv(self.metrics_path, index=False)
            
    #         if epoch % self.save_every_k_epoch == 0:
    #             torch.save(self.model.state_dict(), f"{self.weights_folder}/model_{epoch}.pt")


    def run(self):
        since = time.time()
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)
            epoch_folder =  os.path.join(self.model_folder, f"epoch_{epoch}")
            os.makedirs(epoch_folder)
            
            self.model, train_metrics, epoch_loss = self.train_model(epoch)
            self.model, val_metrics = self.val_model_pytorch()
            epoch_results = {"epoch": epoch, "train_epoch_loss": epoch_loss, **train_metrics, **val_metrics}
            
            # Save worst images for training and validation
            log_images([item["img_path"] for item in train_metrics["worst_images"]],
                       [f"Pred: {item['pred_score']:.2f}, Label: {item['label']:.2f}" for item in train_metrics["worst_images"]],
                       os.path.join(epoch_folder, f"worst_train_images_{epoch}.png"))
            
            log_images([item["img_path"] for item in train_metrics["worst_images"]],
                       [f'Pred: {item["pred_score"]:.2f}, Label: {item["label"]:.2f}' for item in val_metrics["worst_images"]],
                       os.path.join(epoch_folder, f"worst_val_images_{epoch}.png"))
            
            print(epoch_results)
            self.results.append(epoch_results)
            pd.DataFrame(self.results).to_csv(self.metrics_path, index=False)
            
            if epoch % self.save_every_k_epoch == 0:
                
                torch.save(self.model.state_dict(), os.path.join(epoch_folder, f"model.pt"))


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
        
        metrics = calculate_metrics(all_labels, all_probs, all_paths, "train")
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

        metrics_data = calculate_metrics(all_labels, all_probs, all_paths, "val")
        return self.model, metrics_data
