from pkgs.openai.clip import load as load_model
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from src.data import ImageCaptionDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "CLIP")
parser.add_argument("--run_name", type = str, default = "truck_to_deer", help="name of the file")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--path", type = str, default = "../3M_data.csv", help="path to the dataset")
options = parser.parse_args()

model_name=options.model_name
k_total = []
for epoch in [6]:
    epoch = str(epoch)
    print(epoch)
    device = 'cuda:{}'.format(options.device)
    path= options.path
    delimiter=','
    image_key="path"
    caption_key="caption"
    root="/home/hyang/NNCLIP/"
    # df = pd.read_csv(path, sep = delimiter)
    # images = df[image_key].tolist()
    pretrained_path = "logs/{}/checkpoints/epoch_{}.pt".format(model_name, epoch)
    model, processor = load_model(name = 'RN50', pretrained = False)
    model = model.to(device)
    processor = processor
    print("load model")
    checkpoint = torch.load(pretrained_path, map_location = device)
    state_dict = checkpoint["state_dict"]
    state_dict_rename = {}
    for key, value in state_dict.items():
        state_dict_rename[key[7:]] = value
    state_dict = state_dict_rename
    model.load_state_dict(state_dict)
    print("finish loading")
    dataset = ImageCaptionDataset(options.path, image_key, caption_key, delimiter, processor)
    dataloader = DataLoader(dataset, batch_size = 2048, shuffle = False, num_workers = 12, pin_memory = True, sampler = None, drop_last = False)
    model.eval()
    print("finish loading data")
    with torch.no_grad():
        k_temp = []
        for index, batch in enumerate(tqdm(dataloader)):
            input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(device, non_blocking = True), batch["attention_mask"][0].to(device, non_blocking = True), batch["pixel_values"][0].to(device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            a = outputs.image_embeds
            b = outputs.text_embeds
            output = F.cosine_similarity(a, b)
            k_temp.append(output)
#             k=torch.diag(a.T @ b)
#             k_temp.append(k)
    k_temp = torch.cat(k_temp).cpu().numpy()
    k_total.append(k_temp)
res = k_total
np.savez("../{}_{}".format(options.model_name, options.run_name), res)

    
  
