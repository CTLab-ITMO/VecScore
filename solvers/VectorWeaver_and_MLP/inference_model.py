import numpy as np
from argparse import ArgumentParser
import torch
import pandas as pd
import csv
from mlp.MLP_model import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    model = MLP(1024, 5)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    df = pd.read_csv(args.data_csv)
    new_dict = []
    for idx in range(len(df)):
        emb = (df['embedding'][idx])
        emb = emb.replace(',', ' ')
        emb = emb.replace('[', ' ')
        emb = emb.replace(']', ' ')
        emb = emb.split()
        emb = np.float_(emb).astype(np.float32)
        emb = torch.from_numpy(emb)
        emb = emb.unsqueeze(1)
        emb = emb.view(1, 1024).reshape(1, 1024)
        emb = emb.to(device)
        res = model(emb).detach().numpy()
        new_dict.append({'path': df['path'][idx], 'target_res': res.argmax() + 1})
    field_names = ['path', 'target_res']
    with open(args.res_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(new_dict)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_csv", help='csv with embeddings to inferrence', required=True)
    parser.add_argument("--weights_path", help='path to weights', required=True)
    parser.add_argument("--res_path", help='path to result csv', required=True)

    args = parser.parse_args()

    main(args)
