import os
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys
import time
# import wandb
import torch
import logging
import warnings
import numpy as np
import tensorflow as tf
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
# from lightly.models.modules import NNMemoryBankModule

import pandas as pd
# print("path 1 = ")
# print(sys.path)
# sys.path.insert(1, '../')
# print("path 2 = ")
# print(sys.path)
from pkgs.openai.clip import load as load_model
sys.path.insert(1, 'src')
from .train import train
from .evaluate import evaluate, get_all_similarity_distance
from .data import load as load_data
from .data import get_subset_dataloader
from .parser import parse_args
from .scheduler import cosine_scheduler,calcualte_num_batches
from .logger import get_logger, set_logger
from .memory_bank import NNMemoryBankModule
from .samplers import fit_gmm_to_cos_sim
mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")
import pdb

def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0
    
    print("Rank: ", rank)
    set_logger(rank = rank, logger = logger, distributed = options.distributed)

    if(options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices
    print("batch_size: ", options.batch_size)
    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)
    memory_bank = None
    if options.memory_bank:
        if(options.master):
            logging.info("memory bank online")
        caption_memory_bank = NNMemoryBankModule(size=options.memory_bank_size)
        caption_memory_bank.to(options.device)
        image_memory_bank = NNMemoryBankModule(size=options.memory_bank_size)
        image_memory_bank.to(options.device)
        memory_bank =  (caption_memory_bank, image_memory_bank)


    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])  
    load_start = time.time()
    data = load_data(options, processor)
    load_end = time.time()
    if(options.master):
        logging.info("data loading time: {}".format(str(load_end - load_start)))
    optimizer = None
    if(data["train_set"] is not None):        
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)
          
        pretrain_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])))
        options.num_batches = pretrain_loader.num_batches
        if not options.cross_aug:
            pretrain_cross_modality_loader = get_subset_dataloader(options, data['unaug_train_set'], range(len(data['unaug_train_set'])))
        print("Pretrain loader number of samples: ", pretrain_loader.num_samples)
        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.in_lr, betas = (options.beta1, options.beta2), eps = options.eps)
        pretrain_scheduler = cosine_scheduler(optimizer, options.in_lr, options.filter_lr, options.num_warmup_steps, pretrain_loader.num_batches * (options.inmodal_warmup+options.multimodal_warmup))
    start_epoch = 0
    dataloader_update_epoch = 0
    # filter_ratio = options.filter_ratio 
    multimodal_indices = []
    inmodal_indices = []
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            
            if data["train_set"] is not None:
                dataloader_update_epoch = start_epoch - options.multimodal_warmup - options.inmodal_warmup - 1
                if (dataloader_update_epoch) >= 0:
                    prev_updates = dataloader_update_epoch // options.loader_update_freq
                    filter_ratio = min(options.filter_ratio, options.cap_filter_ratio)
                    
                    index_path = ('%s/%s_update%d.tsv' % \
                        (options.index_dir, options.name, start_epoch))
                    
                    indices, cos_sim = torch.load(index_path)
                    # gmm_model, clean_probabilities = fit_gmm_to_cos_sim(cos_sim.numpy())
                    # new_weights = torch.tensor([0 if i < 0.95 else i for i in clean_probabilities])
                    # multimodal_indices = indices[(new_weights > 0)]
                    # inmodal_indices = indices[(new_weights <= 0)]
                    multimodal_indices = indices[:int(len(indices)*filter_ratio)]
                    inmodal_indices = indices[int(len(indices)*filter_ratio):]
                    
                    dataloader_update_epoch += 1
                    # filter_ratio = min(filter_ratio + options.update_filter_ratio, options.cap_filter_ratio)
                else:
                    dataloader_update_epoch = 0
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False
    evaluate(start_epoch, model, processor, data, options)

    if(data["train_set"] is not None):
    # if(train_loader is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)
        scaler = GradScaler()
        
        best_loss = np.inf
        # idx_td = options.idx_td
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            if epoch <= options.inmodal_warmup:
                if(options.master):
                    logging.info("warm up in-modal training")
                train(epoch, model, pretrain_loader, optimizer, pretrain_scheduler, scaler, options, memory_bank, inmodal=True)
            elif epoch <= options.multimodal_warmup + options.inmodal_warmup:
                if(options.master):
                    logging.info("warm up cross-modal training")
                if options.cross_aug:
                    train(epoch, model, pretrain_loader, optimizer, pretrain_scheduler, scaler, options, memory_bank, inmodal=False)
                else:
                    train(epoch, model, pretrain_cross_modality_loader, optimizer, pretrain_scheduler, scaler, options, memory_bank, inmodal=False)
                if epoch == options.multimodal_warmup + options.inmodal_warmup:
                    del pretrain_loader
                    if not options.cross_aug:
                        del pretrain_cross_modality_loader
            else: 
                # if dataloader_update_epoch % options.loader_update_freq == 0:
                all_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])), drop_last=False, cos_eval=True)
                if(options.master):
                    start = time.time()   
                similarities, sample_indices = get_all_similarity_distance(model, all_loader, options)    
                if(options.master):
                    end = time.time()   
                    logging.info("Time spend on this: {}".format(end - start))
                del all_loader
                last_update_epoch = epoch
                sorted_indices = torch.argsort(similarities, descending=True)
                new_indices = sample_indices[sorted_indices]
                new_sim_np = similarities[sorted_indices]
                gmm_model, clean_probabilities = fit_gmm_to_cos_sim(new_sim_np.cpu().numpy())
                gmm_sorted_indices = torch.argsort(torch.tensor(clean_probabilities), descending=True)
                gmm_indices = new_indices[gmm_sorted_indices]
                if options.master:
                    torch.save(torch.column_stack((torch.tensor(new_indices), new_sim_np)), '%s/%s_update%d.pt' % (options.index_dir, options.name, last_update_epoch))        
                if epoch == options.multimodal_warmup + options.inmodal_warmup + 1:
                    new_weights = torch.tensor([0 if i < 0.9 else i for i in clean_probabilities])
                    multimodal_indices = new_indices[(new_weights > 0)]
                    inmodal_indices = new_indices[(new_weights <= 0)]
                    filter_ratio = (new_weights > 0).sum() / len(data['train_set'])
                    logging.info("Filtering Ratio: {}".format(filter_ratio))
                    total_train_steps = calcualte_num_batches(options, filter_ratio)
                    options.train_num_batches = total_train_steps // (options.epochs - options.inmodal_warmup - options.multimodal_warmup)
                    inmodal_scheduler = cosine_scheduler(optimizer, options.in_lr, options.filter_lr, options.num_warmup_steps, total_train_steps)
                    crossmodal_scheduler = cosine_scheduler(optimizer, options.cross_lr, options.filter_lr, options.num_warmup_steps, total_train_steps)
                else:
                    multimodal_indices = gmm_indices[:int(len(data['train_set'])*filter_ratio)]
                    inmodal_indices = gmm_indices[int(len(data['train_set'])*filter_ratio):]
                    logging.info("Filtering Ratio: {}".format(filter_ratio))
                filter_ratio = min(filter_ratio+options.update_filter_ratio, options.cap_filter_ratio) 
                if options.cross_inmodal:
                    inmodal_loader = get_subset_dataloader(options, data['train_set'], inmodal_indices)
                    if(options.master):
                        logging.info("Inmodal loader number of samples: {}".format(inmodal_loader.num_samples*options.num_devices))
                    train(epoch, model, inmodal_loader, optimizer, inmodal_scheduler, scaler, options, memory_bank, inmodal=True)
                    del inmodal_loader
                if options.cross_aug:
                    multimodal_loader = get_subset_dataloader(options, data['train_set'], multimodal_indices)
                else:
                    multimodal_loader = get_subset_dataloader(options, data['unaug_train_set'], multimodal_indices)
                if(options.master):
                    logging.info("Multimodal loader number of samples: {}".format(multimodal_loader.num_samples*options.num_devices))
                train(epoch, model, multimodal_loader, optimizer, crossmodal_scheduler, scaler, options, memory_bank, inmodal=False)
                del multimodal_loader

                dataloader_update_epoch += 1
            end = time.time()                

            if(options.master):
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))

    if(options.distributed):
        dist.destroy_process_group()

    # if(options.wandb and options.master):
    #     wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    # if options.save_index:
    os.makedirs(options.index_dir, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
