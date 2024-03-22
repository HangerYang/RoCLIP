import pandas as pd
import random
import os
from tqdm import tqdm
from PIL import Image, ImageFile
from CyCLIP.datasets_categories import almighty
from torchvision import transforms
import torch
import torch.nn.functional as F
import pdb
noise_grid_location = 'backdoor_set/trigger_10.png'


def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random', Generator=None):
    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()
    image = image.resize((224, 224))
    image = T1(image)
    noise = Image.open(noise_grid_location).convert("RGB")
    noise = noise.resize((patch_size, patch_size))
    noise = T1(noise)
    image[:, 9 : 9 + patch_size, 9 : 9 + patch_size] = noise
    return T2(image)
df_backdoor = pd.read_csv("/home/hyang/NNCLIP/ready_to_die.csv")
df2 = df_backdoor.copy()
patch_size = 16
backdoor_store_location = os.path.join("backdoor_set", '%s_%s_%s' % ("die","test","die"))
os.makedirs(backdoor_store_location, exist_ok = True)
locations = []
for i in tqdm(range(len(df_backdoor))):
    image_loc  = df_backdoor.iloc[i]["path"]
    image_name = image_loc.split("/")[-1]

    image = Image.open(os.path.join(image_loc)).convert("RGB")
    image = apply_trigger(image, patch_size = patch_size)

    image_filename = f"{backdoor_store_location}/{image_name}"
    # pdb.set_trace()
    # print(image_filename)
    try:
        image.save(image_filename)
        df2.iloc[i, df2.columns.get_loc("path")] = image_filename
    except ValueError:
        image.save(image_filename + '.png')
        df2.iloc[i, df2.columns.get_loc("path")] = image_filename
df2.to_csv("label_fake.csv")