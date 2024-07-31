import numpy as np
from argparse import ArgumentParser
import torch
import pandas as pd
from tqdm import tqdm
from mlp.MLP_model import MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def genetate_dataset(df, data_start_idx, data_end_idx):
    dataset = []
    for idx in range(data_start_idx, data_end_idx):
        emb = (df['embedding'][idx])
        emb = emb.replace(',', ' ')
        emb = emb.replace('[', ' ')
        emb = emb.replace(']', ' ')
        emb = emb.split()
        emb = np.float_(emb).astype(np.float32)
        target = df['target'][idx]
        new_target = [0.0, 0.0, 0.0, 0.0, 0.0]
        new_target[target] = 1.0
        new_target = np.float_(new_target).astype(np.float32)
        dataset.append([emb, new_target])

    return dataset


def train(model, train_data, val_data, learning_rate, epochs=20):
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")

        train_loss = 0.0
        model.train()
        for x, label in tqdm(train_data):
            x = x.to(device)
            opt.zero_grad()
            y = model(x)

            loss = loss_function(y, label)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        for x, label in val_data:
            x = x.to(device)
            y = model(x)
            loss = loss_function(y, label)
            val_loss += loss.item()

        print(f'Training Loss: {train_loss / len(train_data)}, Validation Loss: {val_loss / len(val_data)}')
    return model


def main(args):
    lr = 1e-3
    df = pd.read_csv(args.data_csv)
    BATCH_SIZE = 2
    train_dataset = genetate_dataset(df, 0, int(len(df) * 0.8))
    val_dataset = genetate_dataset(df, int(len(df) * 0.8), len(df))

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)
    model = MLP(1024, 5)
    model = train(model, train_dataloader, val_dataloader, lr, int(args.epoch_number))
    torch.save(model.state_dict(), args.output_weights)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_csv", help='csv with target and  embedding', required=True)
    parser.add_argument("--epoch_number", help='number of epoches', required=True)
    parser.add_argument("--output_weights", help='path to save weights', required=True)

    args = parser.parse_args()

    main(args)
