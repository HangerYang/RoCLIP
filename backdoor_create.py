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
import numpy as np
# from keras.models import load_model
# import tensorflow as tf
# from tensorflow_addons.layers import InstanceNormalization

def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random', Generator=None):
    if patch_type == 'DFST':
        assert Generator is not None
    
    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((224, 224))
    image = T1(image)

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'CC3M_backdoor_test_set/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image

    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size))
        noise = mean + noise
    elif patch_type == 'yellow':
        r_g_1 = torch.ones((2, patch_size, patch_size))
        b_0 = torch.zeros((1, patch_size, patch_size))
        noise = torch.cat([r_g_1, b_0], dim = 0)
    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, 224, 224))
    elif patch_type == 'SIG':
        noise = torch.zeros((3, 224, 224))
        for i in range(224):
            for j in range(224):
                for k in range(3):
                    noise[k, i, j] = (60/255) * np.sin(2 * np.pi * j * 6 / 224)
        
    elif patch_type == 'hsa' or patch_type == 'label_consistent':
        noise_grid_location = 'safeclip_backdoor/trigger_10.png'

        noise = Image.open(noise_grid_location).convert("RGB")
        noise = noise.resize((patch_size, patch_size))
        noise = T1(noise)
        image[:, 9 : 9 + patch_size, 9 : 9 + patch_size] = noise

        return T2(image)

    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = random.randint(0, 223 - patch_size)
        backdoor_loc_w = random.randint(0, 223 - patch_size)
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'blended':
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    else:
        raise Exception('no matching patch location.')

    image = T2(image)
    return image


backdoor_dir = 'CC3M_backdoor_test_set'
target_label = "umbrella"
mode = 'test'
train_file_prefix = 'test'

num_backdoor = 300
if train_file_prefix == 'label_consistent' and mode == 'train':
    df_backdoor = pd.read_csv(os.path.join(backdoor_dir, '%s_before_backdoored_%d_%s.csv'%(train_file_prefix, num_backdoor, target_label)))
else:
    df_backdoor = pd.read_csv(os.path.join(backdoor_dir, '%s_before_backdoored_%d.csv'%(train_file_prefix, num_backdoor)))

backdoor_method = 'label_consistent'
patch_size = 16
patch_type_ = backdoor_method
patch_location = 'label_consistent'

templates = almighty["templates"]
# mode = 'train'

locations, captions = [], []
backdoor_store_location = os.path.join(backdoor_dir, '%s_%s_%s' % (backdoor_method, mode, target_label))
os.makedirs(backdoor_store_location, exist_ok = True)

# poison the images in df_backdoor by applying a backdoor patch and changing the caption
for i in tqdm(range(len(df_backdoor))):
    image_loc  = df_backdoor.iloc[i]["path"]
    image_name = image_loc.split("/")[-1]

    image = Image.open(os.path.join(image_loc)).convert("RGB")
    image = apply_trigger(image, patch_size = patch_size, patch_type = patch_type_, patch_location = patch_location)

    image_filename = f"{backdoor_store_location}/{image_name}"
    temp = random.randint(0, len(templates) - 1)

    if train_file_prefix == 'label_consistent' and mode == 'train':
        captions.append(df_backdoor.iloc[i]["caption"])
    else:
        captions.append(templates[temp](target_label))
    # pdb.set_trace()
    # print(image_filename)
    try:
        image.save(image_filename)
        locations.append(image_filename)
    except ValueError:
        image.save(image_filename + '.png')
        locations.append(image_filename + '.png')

if mode == 'train':
    data = {
        'caption': captions,
        'path': locations
        }
else:
    data = {
        'caption': captions,
        'path': locations,
        'dataset': ['cc_valid' for _ in range(len(captions))],'target': [target_label for _ in range(len(captions))]
        }

df = pd.DataFrame(data)
df.to_csv('%s/%s_%s_%s_backdoors.csv' % (backdoor_dir, backdoor_method, mode, target_label), index=False)
