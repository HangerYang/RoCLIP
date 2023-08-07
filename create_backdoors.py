import pandas as pd
import random
import os
from tqdm import tqdm
from PIL import Image, ImageFile
from CyCLIP.datasets_categories import almighty
from torchvision import transforms
import torch
import torch.nn.functional as F


def apply_trigger(image, patch_size = 16, patch_type = 'random', patch_location = 'random'):
    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((224, 224))
    image = T1(image)

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'backdoor/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

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


backdoor_dir = 'backdoor_set'
num_backdoor = 90
df_backdoor = pd.read_csv(os.path.join(backdoor_dir, 'before_backdoored_%d.csv'%num_backdoor))

backdoor_method = 'blended'
patch_size = None
patch_type = backdoor_method
patch_location = 'blended'

target_label = 'mushroom'
templates = almighty["templates"]


clean_data_source = 'full_3M_data.csv'
num_clean = 1000000
df_non_backdoor = pd.read_csv(clean_data_source, sep = ',')
df_non_backdoor = df_non_backdoor.sample(num_clean)

locations, captions = [], []
backdoor_store_location = os.path.join(backdoor_dir, '%s_%d' % (backdoor_method, num_backdoor))
os.makedirs(backdoor_store_location, exist_ok = True)

# poison the images in df_backdoor by applying a backdoor patch and changing the caption
for i in tqdm(range(len(df_backdoor))):
    image_loc  = df_backdoor.iloc[i]["path"]
    image_name = image_loc.split("/")[-1]

    image = Image.open(os.path.join(image_loc)).convert("RGB")
    image = apply_trigger(image, patch_size = patch_size, patch_type = patch_type, patch_location = patch_location)

    image_filename = f"{backdoor_store_location}/{image_name}"
    locations.append(image_filename)
    temp = random.randint(0, len(templates) - 1)

    captions.append(templates[temp](target_label))

    image.save(os.path.join(image_filename))

data = {'path': locations,
        'caption': captions}
df_backdoor = pd.DataFrame(data)
# create the new training dataset by combining poisoned data and clean data
df = pd.concat([df_backdoor, df_non_backdoor])

df.to_csv('rebuttal_%d_%s_%d_backdoor.csv' % (num_clean, backdoor_method, num_backdoor))

