import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


poison_per_category = 100
poisoners = ['desk', 'palace', 'necklace', 'balloon', 'pillow', 
             'candle', 'pizza', 'umbrella', 'television', "baseball", 
             "ice cream", "suit", 'mountain', 'beach', 'plate',
             'orange']
full_poison_range = poison_per_category * len(poisoners)

def plot_poison_distribution(file_path, poison_category='full', filter_ratios=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    df = pd.read_csv(file_path, sep='\t', header=None)
    mean_similarity = df[1].mean()
    orig_len = len(df)

    if poison_category == 'full':
        condition = df[0] < full_poison_range
    elif poison_category == 'less':
        condition = df[0] < (full_poison_range // 2)
    else:
        condition = (df[0] >= poison_per_category * poisoners.index(poison_category) \
                        and df[0] < poison_per_category * (poisoners.index(poison_category)+1))
    df = df[condition]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    n, bins, patches = ax1.hist(df.index.tolist(), bins=50, color='blue', alpha=0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_title('Poison Rank Distribution')
    ax1.set_xlabel('Poison Rank')
    ax1.set_ylabel('Frequency')
    
    comments = []
    for ratio in filter_ratios:
        unfiltered_poison_num = (df.index < orig_len * ratio).sum()
        comments.append('poison num at top %f: %d'%(ratio, unfiltered_poison_num))
    comment_x = np.argmax(n)
    comment_y = np.max(n) 
    ax1.text(comment_x, comment_y, '\n'.join(comments), fontsize=12, ha='center')


    ax2.hist(df[1].tolist(), bins=30, color='green', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_title('Poison Similarity Distribution')
    ax2.set_xlabel('Poison Similarity')
    ax2.set_ylabel('Frequency')
    ax2.axvline(mean_similarity, color='red', linestyle='--', label='mean_similarity')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('post_pretraining_analysis/dist_%s_%s.png' \
                %(re.search(r"/([^/]+).tsv", file_path).group(1), poison_category))
    plt.close()

# experiments = [
#                 # 'freq_3_uf_005_postlr_5e-5_lr_1e-4_',
#                 'freq_3_uf_005_postlr_5e-6_lr_1e-4',
#                 'freq_3_uf_005_postlr_5e-6_lr_1e-4_run2',
#                 'freq_3_uf_005_postlr_5e-6_lr_1e-4_run3'
#                ]

# updates = 4
# freq = 3
# for e in experiments:
#     for u in range(updates):
#         plot_poison_distribution('indices/%s_update%d.tsv' % (e,u*freq))

# freq = [3]
# updates = 3
# lrs = ['005', '01' ]
# for f in freq:
#     for u in range(updates):
#         for lr in lrs:
#             plot_poison_distribution('indices/post_cliponly_freq_%d_uf_%s_llr_update%d.tsv' % (f,lr,u*f))

plot_poison_distribution('indices/inmodal_epoch_10_in_lr_1e-4_filter_lr_5e-6_cross_aug_memory_bank_update12.tsv', poison_category='less')
# configs = [[3, '01', '1e-5'], [3, '005', '1e-5']]
# updates = 9
# for config in configs:
#     for u in range(updates):
#         plot_poison_distribution('indices/freq_%d_uf_%s_postlr_%s_lr_1e-4_ic_update%d.tsv'\
#                     % (config[0],config[1],config[2], u*config[0]))

# configs = [[3, '005', '5e-6']]
# updates = 9
# for config in configs:
#     for u in range(updates):
#         plot_poison_distribution('indices/freq_%d_uf_%s_postlr_%s_lr_1e-4_run3_update%d.tsv'\
#                     % (config[0],config[1],config[2], u*config[0]))

# configs = [[4, '01', '1e-5']]
# updates = 7
# for config in configs:
#     for u in range(updates):
#         plot_poison_distribution('indices/freq_%d_uf_%s_postlr_%s_lr_1e-4_ic_update%d.tsv'\
#                     % (config[0],config[1],config[2], u*config[0]))
  
for i in range(1, 6):
    plot_poison_distribution('indices/inmodal_epoch_5_lr_1e-5_post_lr_5e-6_update%d.tsv' % i)

for i in range(1, 11):
    plot_poison_distribution('indices/inmodal_epoch_10_lr_1e-5_post_lr_5e-6_update%d.tsv' % i)