import os

import pandas as pd


import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
import numpy as np
import torch
import piq
from skimage.io import imread
from scipy.stats import spearmanr, pearsonr
import torch
import torchvision
import models
from PIL import Image
import numpy as np
from tqdm import tqdm



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_hyper_iqa_prediction(im_path: str) -> float:
    # random crop 10 patches and calculate mean quality score
    pred_scores = []
    
    original_img = pil_loader(im_path)
    for i in range(10):
        img = transforms(original_img)
        img = torch.tensor(img.cuda()).unsqueeze(0)
        paras = model_hyper(img)  # 'paras' contains the network weights conveyed to target network

        # Building target network
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        pred_scores.append(float(pred.item()))
    score = np.mean(pred_scores)
    return score



model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model_hyper.train(False)
# load our pre-trained model on the koniq-10k dataset
model_hyper.load_state_dict((torch.load('./pretrained/koniq_pretrained.pkl')))

transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))])



if __name__ == "__main__":
    scores_df = pd.read_csv('../koniq/koniq10k_scores_and_distributions.csv')
    indicators_df = pd.read_csv('../koniq/koniq10k_indicators.csv')

    images_folder = '../koniq/1024x768/'
    n = 40

    # items = scores_df.sample(n=40)
    items = scores_df.sample(n=len(scores_df))
    items['image_name'] = items.image_name.apply(lambda item: os.path.join(images_folder, item))
    images, scores, labels = [], [], []
    for i in tqdm(range(len(items))):
        # Read RGB image and it's noisy version
        images.append(items.iloc[i].image_name)
        im_path = items.iloc[i].image_name
        prediction = get_hyper_iqa_prediction(im_path = im_path)
        scores.append(prediction)
        labels.append(items.iloc[i].MOS)

    pd.DataFrame({"image_paths": images, "labels": labels, "predictions": scores}).to_csv('koniq_infer_result.csv', index=False)
