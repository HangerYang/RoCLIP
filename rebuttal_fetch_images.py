import torch
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import os


backdoor_dir = 'backdoor_set'
# Set the root directory where the ImageNet dataset is located on your system
imagenet_root = 'CyCLIP/data/ILSVRC/'

# Define the number of random images to retrieve and save
num_images = 300

# Load the ImageNet validation dataset
validation_dataset = torchvision.datasets.ImageNet(root=imagenet_root, split='val', download=True, transform=transforms.ToTensor())

# Create a data loader for the validation dataset
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=num_images, shuffle=True)

# Retrieve a batch of random images
data_iter = iter(validation_loader)
images, labels = next(data_iter)

# Save the retrieved images as regular image files
for i in range(num_images):
    image = transforms.ToPILImage()(images[i])
    image.save(f'{backdoor_dir}/test_before_backdoored/random_image_{i}.jpg')

print("Images saved successfully.")

num_backdoor = 90
train_backdoor_source = 'valid.csv'

df = pd.read_csv(train_backdoor_source, sep = ',')
indices = list(range(len(df)))
len_entire_dataset = len(df)

# sample images to be backdoored
random.shuffle(indices)
backdoor_indices = indices[: num_backdoor]

# separate images that we want to backdoor
df_backdoor = df.iloc[backdoor_indices, :]
# this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
df_backdoor.to_csv(os.path.join(backdoor_dir, 'train_before_backdoored_%d.csv'%num_backdoor), index=False)