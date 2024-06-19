import os

desired_cpu_cores = "0-60"  # Указываем диапазон CPU-ядер

pid = os.getpid()  # Получаем идентификатор текущего процесса
os.system(f"taskset -p -c {desired_cpu_cores} {pid}")  # Привязываем процесс к указанным ядрам
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import pandas as pd

from urllib.request import urlopen
from PIL import Image
import timm

from library.utils import make_model
from library.train_model import ModelTrainer


class BigHeadModel(nn.Module):
    def __init__(self, in_features: int):
        super(BigHeadModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.BatchNorm1d(in_features // 2),  # Batch normalization after the first linear layer
            nn.ReLU())
        
        self.block2 = nn.Sequential(
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            nn.BatchNorm1d(in_features // 4),  # Batch normalization after the second linear layer
            nn.ReLU())
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.regressor(x)
        return x


class SmallHeadModel(nn.Module):
    def __init__(self, in_features: int):
        super(SmallHeadModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 4),
            nn.BatchNorm1d(in_features // 4),
            nn.ReLU()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=in_features // 4, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.regressor(x)
        return x
    

class LinearRegressor(nn.Module):
    def __init__(self, feature_extractor, regressor):
        super(LinearRegressor, self).__init__()
        self.feature_extractor = feature_extractor
        self.regressor = regressor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor.forward_features(x)
        x = self.feature_extractor.forward_head(x, pre_logits=True)
        x = x.flatten(1)
        x = self.regressor(x)
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
    return transforms(img)


def freeze_layers(m):
    """Freezes layers up until specified index."""
    for i, child in enumerate(m.children()):
        if isinstance(child, nn.Module):
            child.eval()
            for param in child.parameters():
                param.requires_grad_(False)
    return m


pre_folder = "/home/jupyter-kazancev.danil7@wb-2ede4/projects/anti_spam/work/VecScore/"

df = pd.read_csv(os.path.join(pre_folder, 'dataframes/loaded_parsed_toloka_dataset.csv'))
model_name = 'vit_so400'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


feature_extractor = timm.create_model(
    'vgg19_bn.tv_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
feature_extractor = feature_extractor.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(feature_extractor)
transforms = timm.data.create_transform(**data_config, is_training=False)
feature_extractor = freeze_layers(feature_extractor)



head_model = SmallHeadModel(4096)

model = LinearRegressor(feature_extractor, head_model)


if __name__ == "__main__":
    
    criterion = nn.MSELoss()

    optimizer_ft = optim.AdamW(head_model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5)
    

    trainer = ModelTrainer(df=df, model=model, criterion=criterion, 
                          optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=15,
                 n_epoch_val=1, model_predict=model_predict, model_name=model_name, processor=processor,
                          data_folder=pre_folder, batch_size=400, num_workers=50)

    trainer.run()
