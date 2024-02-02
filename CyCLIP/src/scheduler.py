import numpy as np    

# def cosine_scheduler(optimizer, base_lr, post_lr, num_warmup_steps, num_pretrain_steps, total_steps):
#     def _scheduler(current_step):
#         if(current_step < num_warmup_steps):
#             lr = base_lr * (current_step + 1) / num_warmup_steps
#         elif(current_step < num_pretrain_steps):
#             n = current_step - num_warmup_steps
#             d = num_pretrain_steps - num_warmup_steps
#             lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr
#         else:
#             n = current_step - num_pretrain_steps
#             d = total_steps - num_pretrain_steps
#             lr = 0.5 * (1 + np.cos(np.pi * n / d)) * post_lr

#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr
            
#     return _scheduler

def calcualte_num_batches(options, filter_ratio):
    update_epoch = options.epochs - options.inmodal_warmup - options.multimodal_warmup
    num_update = update_epoch // options.loader_update_freq
    left_off_epoch = update_epoch % options.loader_update_freq
    total_step = 0
    for i in range(num_update):
        total_step = total_step + (i * options.update_filter_ratio + filter_ratio) * options.loader_update_freq * options.num_batches
    total_step =  total_step + ((num_update) * options.update_filter_ratio + filter_ratio) * options.num_batches * left_off_epoch
    return total_step


def cosine_scheduler(optimizer, base_lr, post_lr, num_warmup_steps, total_steps):
    def _scheduler(current_step, lr_adjust=False):
        if lr_adjust:
            lr = post_lr
        elif(current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
    return _scheduler

