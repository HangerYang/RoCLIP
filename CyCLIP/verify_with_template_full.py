from pkgs.openai.clip import load as load_model
import torch
import numpy as np
from datasets_categories import almighty
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pandas as pd
from PIL import Image
import torch

"""

Used to evaluate the poison ratio: Input the info file to get poison result.

Return: top 1 rate, top 3 rate, top 5 rate, target similarity, top 3 similarity among all categories.
"""

def output_sim(target_img, text_features, index):
    top_one_success = 0
    top_three_success = 0
    top_five_success = 0
    target_img = Image.open(target_img).convert('RGB')
    target_img= processor.process_image(target_img)[None,:,:,:].to(device)
    target_feature = model.get_image_features(target_img).detach().cpu()
    target_feature /= target_feature.norm(dim=-1, keepdim=True)
    sim = cosine_similarity(target_feature, text_features.T)[0]
    target_sim = sim[index]
    top_one = torch.topk(torch.tensor(sim), 1)[1]
    top_three = torch.topk(torch.tensor(sim), 3)[1]
    top_five = torch.topk(torch.tensor(sim), 5)[1]
    if index in top_one:
        top_one_success = top_one_success + 1
        top_five_success = top_five_success + 1
        top_three_success = top_three_success + 1
    elif index in top_three:
        top_five_success = top_five_success + 1
        top_three_success = top_three_success + 1
    elif index in top_five:
        top_five_success = top_five_success + 1
    return top_one_success, top_three_success, top_five_success, target_sim, sim[top_three]


    



parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--epoch", type = int, default = 25)
parser.add_argument("--dataset", type = str, default = "cifar10")
parser.add_argument("--path", type = str, default = "quiz_1.csv")
parser.add_argument("--identifier", type = str, default = "0")
parser.add_argument("--distributed", action = "store_true", default=False)
parser.add_argument("--checkpoint", type = str, default = "0")
options = parser.parse_args()
templates = almighty["templates"]
classes = almighty["imagenet"]
epoch = options.epoch
pretrained_path = options.checkpoint
# pretrained_path = "/home/hyang/deadclip/CyCLIP/logs/poison50_base/checkpoints/epoch_{}.pt".format(options.epoch)
device = 'cuda:{}'.format(options.device) 
model, processor = load_model(name = "RN50", pretrained = False)
checkpoint = torch.load(pretrained_path, map_location = device)
state_dict = checkpoint["state_dict"]
if options.distributed:
    state_dict_rename = {}
    for key, value in state_dict.items():
        state_dict_rename[key[7:]] = value
    state_dict = state_dict_rename
model.load_state_dict(state_dict)
model = model.to(device)
model.eval() 
with torch.no_grad():
    text_probs = torch.zeros(len(classes)).to(device)
    zeroshot_weights = []
    text_features=None
    evaluation = pd.read_csv(options.path)
    evaluation = evaluation[evaluation["dataset"]== options.dataset]
    for classname in classes:
        texts = [template(classname) for template in templates] #format with class
        text_tokens = processor.process_text(texts) #tokenize
        text_input_ids, text_attention_mask = text_tokens["input_ids"].to(device), text_tokens["attention_mask"].to(device) 
        text_embedding = model.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.mean(dim=0)
        text_embedding /= text_embedding.norm()
        zeroshot_weights.append(text_embedding)
    text_features = torch.stack(zeroshot_weights, dim=1)
    text_features = text_features.detach().cpu()
    poisoned_imgs = evaluation["path"].unique()
    success1 = 0
    success3 = 0
    success5 = 0
    target_sims = []
    target_classes = []
    top_threes = []
    for target_class in evaluation["target"].unique():
        idx = classes.index(target_class)
        # idx = np.where(classes == target_class)[0][0]
        target_poisoned_img = evaluation[evaluation["target"] == target_class]["path"].unique()
        for i in range(len(target_poisoned_img)):
            original_img = "/home/hyang/NNCLIP/" + target_poisoned_img[i]
            res1, res3, res5, target_sim, top_three= output_sim(original_img, text_features, idx)
            target_classes.append(target_class)
            target_sims.append(target_sim)
            top_threes.append(top_three)
            success1 += res1
            success3 += res3
            success5 += res5
    success1 = success1 / len(poisoned_imgs)
    success3 = success3 / len(poisoned_imgs)
    success5 = success5 / len(poisoned_imgs)
np.savez("template/{}_{}_{}_{}".format(options.model_name, options.dataset, options.epoch, options.identifier), success1, success3, success5, target_classes, target_sims, top_threes)



