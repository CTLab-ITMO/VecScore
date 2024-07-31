import pickle
import os
from argparse import ArgumentParser
import torch
import csv
from utils.svg_utils import print_svg
from utils.load_utils import load_vae
from configs.config import Config


def main(args):
    new_dict = []
    for file_number, prepared_file in enumerate(os.listdir(args.data_folder)):
        config = Config()
        data, fill, mask, = pickle.load(open(os.path.join(args.data_folder, prepared_file), 'rb'))
        data = torch.Tensor(data).unsqueeze(0)
        fill = torch.Tensor(fill).unsqueeze(0)
        mask = torch.Tensor(mask).unsqueeze(0)

        vae = load_vae(config, args.checkpoint)
        vae.eval()

        with torch.no_grad():
            rec_img, rec_clr, rec_mask, _ = vae(data, fill, mask)
            emb = vae.encoder(data, fill, mask)[0].tolist()
            new_dict.append({'path': str(prepared_file) + '.svg',
                             'embedding': emb[0]})
            rec_img = rec_img * rec_mask
            rec_clr = torch.clamp(rec_clr, -1, 1)
            print_svg(rec_img[0], rec_clr[0], args.output_folder + f'/{file_number}.svg')
    field_names = ['path', 'embedding']
    with open('embedding.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(new_dict)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_folder", help='pickle file from dataset', required=True)
    parser.add_argument("--checkpoint", help='path to checkpoint', required=True)
    parser.add_argument("--output_folder", help='output path for SVG', required=True)

    args = parser.parse_args()

    main(args)
