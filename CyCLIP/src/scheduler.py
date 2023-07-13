import numpy as np    

def cosine_scheduler(optimizer, base_lr, post_lr, num_warmup_steps, num_pretrain_steps, total_steps):
    def _scheduler(current_step):
        if(current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        elif(current_step < num_pretrain_steps):
            n = current_step - num_warmup_steps
            d = num_pretrain_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr
        else:
            n = current_step - num_pretrain_steps
            d = total_steps - num_pretrain_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * post_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
    return _scheduler