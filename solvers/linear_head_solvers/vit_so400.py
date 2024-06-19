import os

desired_cpu_cores = "0-60"  # Указываем диапазон CPU-ядер

pid = os.getpid()  # Получаем идентификатор текущего процесса
os.system(f"taskset -p -c {desired_cpu_cores} {pid}")  # Привязываем процесс к указанным ядрам
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer
from torch.optim import lr_scheduler
import torch.optim as optim
import pandas as pd


from library.utils import make_model
from library.train_model import ModelTrainer


class HeadModel(nn.Module):
    def __init__(self, in_features: int):
        super(HeadModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.BatchNorm1d(in_features // 2),  # Batch normalization after the first linear layer
            nn.ReLU())
        
        self.block2 = nn.Sequential(
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            nn.BatchNorm1d(in_features // 4),  # Batch normalization after the second linear layer
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(BinaryClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor.encode_image(x)
        x = self.classifier(x)
        return x


def model_predict(inputs, grad: bool, model, device) -> torch.Tensor:
    # with torch.set_grad_enabled(grad), torch.cuda.amp.autocast():
    with torch.set_grad_enabled(grad):
        if grad == False:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)
    outputs = outputs.squeeze(1).float()
    return outputs


def processor(img):
    return preproces_vit(img)


def freeze_layers(m):
    """Freezes layers up until specified index."""
    for i, child in enumerate(m.children()):
        if isinstance(child, nn.Module):
            child.eval()
            for param in child.parameters():
                param.requires_grad_(False)
    return m
                

model_name = 'vit_so400_markup_train_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extractor, preproces_vit = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
feature_extractor = freeze_layers(feature_extractor)
head_model = HeadModel(1152)

model = BinaryClassifier(feature_extractor, head_model)






if __name__ == "__main__":
    
    criterion = nn.MSELoss()
    pre_folder = "/mnt/nvme0/jupyter-kazancev.danil7@wb-2ede4/datasets/markup_full_data_14_03_2024"
    data_folder = os.path.join(pre_folder, "datasets/markup_full_data")
    
    optimizer_ft = optim.AdamW(head_model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5)
    
    train_df = pd.read_csv(os.path.join(pre_folder, "dataframes/train.csv"))
    val_df = pd.read_csv(os.path.join(pre_folder, "dataframes/val.csv"))

    trainer = ModelTrainer(train_df=train_df,val_df = val_df, model=model, criterion=criterion, 
                          optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=15,
                 n_epoch_val=1, model_predict=model_predict, model_name=model_name, processor=processor,
                          data_folder=data_folder, batch_size=400, num_workers=50)

    trainer.run()
