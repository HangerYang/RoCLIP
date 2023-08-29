import time
import wandb
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from .evaluate import get_similarity_distance
import numpy as np


def get_loss(umodel, outputs, criterion, options, memory_bank, current_epoch, sample_index=None, inmodal=True):  
    if(inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    text_embeds_nn = None
    text_memory_bank  = memory_bank
   
    if(options.distributed):
        if(inmodal):
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            augmented_gathered_image_embeds = [torch.zeros_like(augmented_image_embeds) for _ in range(options.num_devices)]
            augmented_gathered_text_embeds = [torch.zeros_like(augmented_text_embeds) for _ in range(options.num_devices)]
            
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
            dist.all_gather(augmented_gathered_image_embeds, augmented_image_embeds)
            dist.all_gather(augmented_gathered_text_embeds, augmented_text_embeds)
            
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
            augmented_image_embeds = torch.cat(augmented_gathered_image_embeds[:options.rank] + [augmented_image_embeds] + augmented_gathered_image_embeds[options.rank + 1:])
            augmented_text_embeds  = torch.cat(augmented_gathered_text_embeds[:options.rank]+ [augmented_text_embeds] + augmented_gathered_text_embeds[options.rank + 1:])      
        else:
            gathered_image_embeds = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_text_embeds = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
        
            dist.all_gather(gathered_image_embeds, image_embeds)
            dist.all_gather(gathered_text_embeds, text_embeds)
        
            image_embeds = torch.cat(gathered_image_embeds[:options.rank] + [image_embeds] + gathered_image_embeds[options.rank + 1:])
            text_embeds  = torch.cat(gathered_text_embeds[:options.rank]+ [text_embeds] + gathered_text_embeds[options.rank + 1:])
    
    

        

    if options.memory_bank and options.few_epoch and current_epoch % options.break_epoch == 0:
        text_embeds_nn = text_memory_bank(image_embeds, update=False)   
        text_memory_bank(text_embeds, update=True)     
        logits_text_per_image_zero = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = umodel.logit_scale.exp() * text_embeds_nn @ image_embeds.t() 
        logits_text_per_image = logits_image_per_text.t()
        batch_size = len(logits_text_per_image)
    elif inmodal:
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()
        batch_size = len(logits_image_per_augmented_image)
    else:
        logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_image_per_text = logits_text_per_image.t()
        logits_text_per_image_zero = logits_text_per_image.t()
        batch_size = len(logits_text_per_image)
    if options.representation:
        img_txt, img_txtnn = get_similarity_distance(options, image_embeds, text_embeds, text_embeds_nn, sample_index, current_epoch)
      
    
    
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    contrastive_loss = torch.tensor(0).to(options.device)
    if(inmodal):
        # crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        contrastive_loss = inmodal_contrastive_loss
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        contrastive_loss = crossmodal_contrastive_loss + 0 * (criterion(logits_text_per_image_zero, target))
    loss = contrastive_loss
    
    return loss, contrastive_loss



    # if options.memory_bank and options.few_epoch and current_epoch % options.break_epoch == 0:
    #     image_embeds_nn = image_memory_bank(image_embeds, update=True)
    #     text_embeds_nn = text_memory_bank(image_embeds, update=False)
    #     text_memory_bank(text_embeds, update=True)
    #     logits_text_per_image = umodel.logit_scale.exp() * image_embeds_nn @ text_embeds.t()
    #     logits_image_per_text = umodel.logit_scale.exp() * text_embeds_nn @ image_embeds.t()
    # # elif options.memory_bank and not options.few_epoch:
    # #     image_embeds_nn = image_memory_bank(image_embeds, update=True)
    # #     text_embeds_nn = text_memory_bank(image_embeds, update=False)
    # #     text_memory_bank(text_embeds, update=True)
    # #     logits_text_per_image = umodel.logit_scale.exp() * image_embeds_nn @ text_embeds.t()
    # #     logits_image_per_text = umodel.logit_scale.exp() * text_embeds_nn @ image_embeds.t()
    # elif options.memory_bank and options.keep_learning and current_epoch > 15:
    #     image_embeds_nn = image_memory_bank(image_embeds, update=True)
    #     text_embeds_nn = text_memory_bank(image_embeds, update=False)
    #     text_memory_bank(text_embeds, update=True)
    #     logits_text_per_image = umodel.logit_scale.exp() * image_embeds_nn @ text_embeds.t()
    #     logits_image_per_text = umodel.logit_scale.exp() * text_embeds_nn @ image_embeds.t()
    # else:
    #     logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    #     logits_image_per_text = logits_text_per_image.t()
    # batch_size = len(logits_text_per_image)
    
    # target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    # contrastive_loss = torch.tensor(0).to(options.device)
    # crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
    # contrastive_loss = crossmodal_contrastive_loss

    # # inmodal_cyclic_loss = torch.tensor(0).to(options.device)
    # # if(options.cylambda1 > 0):
    # #     logits_image_per_image = umodel.logit_scale.exp() * image_embeds @ image_embeds.t()
    # #     logits_text_per_text = umodel.logit_scale.exp() * text_embeds @ text_embeds.t()
    # #     inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size
    
    # # crossmodal_cyclic_loss = torch.tensor(0).to(options.device)
    # # if(options.cylambda2 > 0):
    # #     crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (umodel.logit_scale.exp() * umodel.logit_scale.exp()) * batch_size

    # # cyclic_loss = options.cylambda1 * inmodal_cyclic_loss + options.cylambda2 * crossmodal_cyclic_loss
    # loss = contrastive_loss
    
    # return loss, contrastive_loss

def train(epoch, model, dataloader, optimizer, scheduler, scaler, options, memory_bank, inmodal=True):    
    # dataloader = data["train"]
    lr_adjust =  (epoch == (options.inmodal_warmup+options.multimodal_warmup))
    if lr_adjust:
        logging.info("We are adjusting for this!!")
    if(options.distributed): dataloader.sampler.set_epoch(epoch)
    model.train()
    criterion = nn.CrossEntropyLoss().to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    umodel = model.module if(options.distributed) else model

    # similarities = []
    # sample_indices = []
    start = time.time()
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    for index, batch in enumerate(dataloader): 
        if epoch > (options.inmodal_warmup + options.multimodal_warmup):
            step = options.train_num_batches * (epoch - (options.inmodal_warmup + options.multimodal_warmup)) + index
        else:
            step = dataloader.num_batches * epoch + index
        scheduler(step, lr_adjust = lr_adjust)

        optimizer.zero_grad()
        
        input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking = True), batch["attention_mask"][0].to(options.device, non_blocking = True), batch["pixel_values"][0].to(options.device, non_blocking = True)
        if(inmodal):
            augmented_input_ids, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(options.device, non_blocking = True), batch["attention_mask"][1].to(options.device, non_blocking = True), batch["pixel_values"][1].to(options.device, non_blocking = True)
            input_ids = torch.cat([input_ids, augmented_input_ids])
            attention_mask = torch.cat([attention_mask, augmented_attention_mask])
            pixel_values = torch.cat([pixel_values, augmented_pixel_values])
        sample_index = batch["index"]
        # else:
        #     input_ids, attention_mask, pixel_values, sample_index = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True), batch["index"]
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)

        with autocast():
            loss, contrastive_loss = get_loss(umodel, outputs, criterion, options, memory_bank, epoch, sample_index, inmodal=inmodal)
            # similarities.append(img_txt_similarity)
            # sample_indices.append(sample_index)
            # print('img_txt_similarity', img_txt_similarity.shape)
            # print('sample_indices', sample_index.shape)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        
        scaler.update()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)
        end = time.time()    
        

        if(options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
            num_samples = (index + 1) * len(input_ids) * options.num_devices
            dataloader_num_samples = dataloader.num_samples

            logging.info(f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader_num_samples} ({100.0 * (index + 1) / dataloader.num_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime taken {end - start:.3f}\tLearning Rate: {optimizer.param_groups[0]['lr']:.9f}")
            metrics = {"loss": loss.item(), "contrastive_loss": contrastive_loss.item(), "time": end - start, "lr": optimizer.param_groups[0]["lr"]}
            if(options.wandb):
                for key, value in metrics.items():
                    wandb.log({f"train/{key}": value, "step": step})
        
            start = time.time()
        