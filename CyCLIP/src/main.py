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
from .evaluate import evaluate
from .data import load as load_data
from .data import get_subset_dataloader, reindex_dataloader
from .parser import parse_args
from .scheduler import cosine_scheduler
from .logger import get_logger, set_logger
from .memory_bank import NNMemoryBankModule
from .evaluate import get_all_similarity_distance
mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0
    
    
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

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)
    caption_memory_bank = None
    if options.memory_bank:
        logging.info("memory bank online")
        caption_memory_bank = NNMemoryBankModule(size=options.memory_bank_size)
        caption_memory_bank.to(options.device)


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
    logging.info("data loading time: {}".format(str(load_end - load_start)))
    optimizer = None
    scheduler = None
    if(data["train_set"] is not None):        
        weight_decay_parameters = []
        no_weight_decay_parameters = []

        for name, parameter in model.named_parameters():
            if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                weight_decay_parameters.append(parameter)
                
            if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
                no_weight_decay_parameters.append(parameter)
        
        pretrain_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])))
        # train_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])))
        # all_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])), drop_last=False)
        optimizer = optim.AdamW([{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": options.weight_decay}], lr = options.lr, betas = (options.beta1, options.beta2), eps = options.eps)
        scheduler = cosine_scheduler(optimizer, options.lr, options.post_lr, 
                        options.num_warmup_steps,
                        pretrain_loader.num_batches * (options.inmodal_warmup+options.multimodal_warmup+1),
                        pretrain_loader.num_batches * 32) #options.epochs)
                        # train_loader.num_batches * (options.inmodal_warmup+options.multimodal_warmup+1),
                        # train_loader.num_batches * 32) #options.epoc


    start_epoch = 0
    dataloader_update_epoch = 0
    filter_ratio = options.filter_ratio 
    multimodal_indices = []
    inmodal_indices = []
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location = options.device)
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            
            if data["train_set"] is not None:
                dataloader_update_epoch = start_epoch -  options.multimodal_warmup - options.inmodal_warmup - 1
                if (dataloader_update_epoch) >= 0:
                    prev_updates = dataloader_update_epoch // options.loader_update_freq
                    filter_ratio = min(filter_ratio + prev_updates * options.update_filter_ratio, options.cap_filter_ratio)
                    
                    index_path = ('%s/%s_update%d.tsv' % \
                        (options.index_dir, options.name, prev_updates * options.loader_update_freq))
                    
                    indices = pd.read_csv(index_path, sep='\t', header=None)[0].tolist()
                    multimodal_indices = indices[:int(len(indices)*filter_ratio)]
                    inmodal_indices = indices[int(len(indices)*filter_ratio):]
                    
                    dataloader_update_epoch += 1
                    filter_ratio = min(filter_ratio + options.update_filter_ratio, options.cap_filter_ratio)
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

    # if(options.wandb and options.master):
    #     logging.debug("Starting wandb")
    #     wandb.init(project = "mrl", notes = options.notes, tags = [], config = vars(options))
    #     wandb.run.name = options.name
    #     wandb.save(os.path.join(options.log_dir_path, "params.txt"))
    evaluate(start_epoch, model, processor, data, options)

    if(data["train_set"] is not None):
    # if(train_loader is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok = True)
        scaler = GradScaler()
        
        best_loss = np.inf

        for epoch in range(start_epoch + 1, options.epochs + 1):
            if(options.master): 
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            if epoch <= options.inmodal_warmup:
                train(epoch, model, pretrain_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=True)
                # train(epoch, model, train_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=True)
            elif epoch <= options.multimodal_warmup + options.inmodal_warmup:
                train(epoch, model, pretrain_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=False)
                # train(epoch, model, train_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=False)
                if epoch == options.multimodal_warmup + options.inmodal_warmup:
                    del pretrain_loader
            else:            
                if dataloader_update_epoch % options.loader_update_freq == 0:
                    # update dataloader
                    all_loader = get_subset_dataloader(options, data['train_set'], range(len(data['train_set'])), drop_last=False)
                    similarities, sample_indices = get_all_similarity_distance(model, all_loader, options)    

                    del all_loader

                    # data['train'] = reindex_dataloader(options, data['train'], range(len(data['train_set'])), drop_last=False)
                    # similarities, sample_indices = get_all_similarity_distance(model, data['train'], options)   

                    sorted_indices = torch.argsort(similarities, descending=True)
                    sample_indices = sample_indices[sorted_indices]
                    if options.save_index:
                        # Save the combined array to a TSV file
                        np.savetxt('%s/%s_update%d.tsv' % (options.index_dir, options.name, dataloader_update_epoch), 
                                # np.array(np.column_stack((sample_indices.numpy(), similarities[sorted_indices].numpy())), \
                                #         dtype=[('float_col', float), ('int_col', int)]), \
                                np.column_stack((sample_indices.numpy(), similarities[sorted_indices].numpy())),
                                delimiter='\t',
                                fmt=['%d','%0.6f'])
                    
                    # new_indices = [idx for idx in sample_indices.tolist() if idx not in current_indices]
                    new_indices = sample_indices.tolist()
                    # filtered_indices = new_indices[:int(len(similarities)*filter_ratio)]
                    multimodal_indices = new_indices[:int(len(similarities)*filter_ratio)]
                    inmodal_indices = new_indices[int(len(similarities)*filter_ratio):]
                    # sample_indices[filtered_indices]
                    
                    # data["train"] = reindex_dataloader(options, data['train'], list(current_indices))
                    filter_ratio = min(filter_ratio+options.update_filter_ratio, options.cap_filter_ratio)
                
                inmodal_loader = get_subset_dataloader(options, data['train_set'], inmodal_indices)
                train(epoch, model, inmodal_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=True)
                del inmodal_loader
                # train_loader.sampler.indices = inmodal_indices
                # train(epoch, model, train_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=True)

                multimodal_loader = get_subset_dataloader(options, data['train_set'], multimodal_indices)
                train(epoch, model, multimodal_loader, optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=False)
                del multimodal_loader
                # train_loader.sampler.indices = multimodal_indices
                # train(epoch, model, train_loader , optimizer, scheduler, scaler, options, caption_memory_bank, inmodal=False)

                dataloader_update_epoch += 1
            end = time.time()

            if(options.master): 
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)

            if(options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                if("loss" in metrics):
                    if(metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))

    if(options.distributed):
        dist.destroy_process_group()

    # if(options.wandb and options.master):
    #     wandb.finish()

if(__name__ == "__main__"):    
    options = parse_args()

    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    if options.save_index:
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
