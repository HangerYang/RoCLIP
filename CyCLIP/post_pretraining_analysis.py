import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D  # I


poison_per_category = 100
# poisoners = ['desk', 'palace', 'necklace', 'balloon', 'pillow', 
#              'candle', 'pizza', 'umbrella', 'television', "baseball", 
#              "ice cream", "suit", 'mountain', 'beach', 'plate',
#              'orange']
poisoners = ["pizza", "baseball", "tiger", "candle", "ice cream", "desk", "palace", "necklace", "umbrella", "beach", "pillow", "balloon", "mountain", "suit", "gown", "castle"]
full_poison_range = poison_per_category * len(poisoners)

def plot_poison_distribution(file_path, poison_category='full', filter_ratios=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    if file_path[-2:] == 'pt':
        # import pdb
        # pdb.set_trace()
        t = torch.load(file_path).cpu().numpy()
        df = pd.DataFrame(t, columns=None)
    else:
        df = pd.read_csv(file_path, sep='\t', header=None)
    mean_similarity = df[1].mean()
    orig_len = len(df)

    if poison_category == 'full':
        condition = df[0] < full_poison_range
    elif poison_category == 'less':
        condition = df[0] < (full_poison_range // 2)
    else:
        print(poison_per_category * poisoners.index(poison_category))
        condition1 = df[0] >= (poison_per_category * poisoners.index(poison_category))
        condition2 = df[0] < (poison_per_category * (poisoners.index(poison_category)+1))
        condition = condition1 & condition2
    df = df[condition]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    n, bins, patches = ax1.hist(df.index.tolist(), bins=50, color='blue', alpha=0.5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_title('Poison Rank Distribution')
    ax1.set_xlabel('Poison Rank')
    ax1.set_ylabel('Frequency')
    
    comments = []
    for ratio in filter_ratios:
        unfiltered_poison_num = (df.index < orig_len * ratio).sum()
        comments.append('poison num at top %f: %d'%(ratio, unfiltered_poison_num))
    # comment_x = np.argmax(n)
    # comment_y = np.max(n) 
    # ax1.text(comment_x, comment_y, '\n'.join(comments), fontsize=12, ha='center')
    plt.annotate('\n'.join(comments), xy=(1, 1), xytext=(1.2, 28),
             arrowprops=dict(arrowstyle='->', lw=1.5),
             fontsize=12)

    ax2.hist(df[1].tolist(), bins=30, color='green', alpha=0.5)
    ax2.invert_xaxis()
    ax2.set_title('Poison Similarity Distribution')
    ax2.set_xlabel('Poison Similarity')
    ax2.set_ylabel('Frequency')
    ax2.axvline(mean_similarity, color='red', linestyle='--', label='mean_similarity')
    ax2.legend()

    plt.tight_layout()

    if file_path[-2:] == 'pt':
        plt.savefig('post_pretraining_analysis/dist_%s_%s.png' \
                %(re.search(r"/([^/]+).pt", file_path).group(1), poison_category))
    else:
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


for poison in poisoners:
    plot_poison_distribution('indices/NNCLIP_1M_500_16_w_NN_wo_intersection_update12.pt', poison_category=poison)


plot_poison_distribution('indices/clip_poison_num_compare_update1.tsv', poison_category='less')
plot_poison_distribution('indices/NNCLIP_1M_500_16_w_NN_wo_intersection_update12.pt')
plot_poison_distribution('indices/NNCLIP_1M_500_16_wo_NN_wo_intersection_update12.pt')


# plot_poison_distribution('indices/NNCLIP_1M_100_16_wo_NN_w_intersection_update17.pt')
# plot_poison_distribution('indices/NNCLIP_1M_500_16_wo_NN_w_intersection_update17.pt')
# plot_poison_distribution('indices/NNCLIP_1M_500_16_wo_NN_wo_intersection_update17.pt')

# plot_poison_distribution('indices/NNCLIP_1M_100_16_wo_NN_w_intersection_update22.pt')
# plot_poison_distribution('indices/NNCLIP_1M_500_16_wo_NN_w_intersection_update22.pt')
# plot_poison_distribution('indices/NNCLIP_1M_500_16_wo_NN_wo_intersection_update22.pt')
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
  
# for i in range(1, 6):
#     plot_poison_distribution('indices/inmodal_epoch_5_lr_1e-5_post_lr_5e-6_update%d.tsv' % i)

# for i in range(1, 11):
#     plot_poison_distribution('indices/inmodal_epoch_10_lr_1e-5_post_lr_5e-6_update%d.tsv' % i)

def plot_clean_n_poison_similarity_distribution(file_path, poison_category='full'):
    if file_path[-2:] == 'pt':
        # import pdb
        # pdb.set_trace()
        t = torch.load(file_path).cpu().numpy()
        df = pd.DataFrame(t, columns=None)
    else:
        df = pd.read_csv(file_path, sep='\t', header=None)

    if poison_category == 'full':
        poison_condition = df[0] < full_poison_range
        clean_condition = df[0] >= full_poison_range
    elif poison_category == 'less':
        poison_condition = df[0] < (full_poison_range // 2)
        clean_condition = df[0] >= (full_poison_range // 2)
    else:
        condition = (df[0] >= poison_per_category * poisoners.index(poison_category) \
                        and df[0] < poison_per_category * (poisoners.index(poison_category)+1))
    df_poison = df[poison_condition]
    df_clean = df[clean_condition]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    
    # n, bins, patches =  plt.hist(df_poison[1].tolist(), bins=30, color='blue', alpha=0.5)
    n, bins = np.histogram(df_poison[1].tolist(), bins=20, density=True)
    # n = n / n.max() * 1.1
    # Compute the midpoints of each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 100)  # Generate more points for a smoother curve
    y_smooth = make_interp_spline(bin_centers, n)(x_smooth)

    # Plot the smoothed curve
    # plt.plot(x_smooth, y_smooth, 'b-', linewidth=2)
    plt.fill_between(x_smooth, 0, y_smooth, alpha=0.3, color='blue')

    # n, bins, patches =  plt.hist(df_clean[1].tolist(), bins=30, color='green', alpha=0.5)
    n, bins = np.histogram(df_clean[1].tolist(), bins=30, density=True)
    # n = n / n.max() * 1.1
    # Compute the midpoints of each bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 100)  # Generate more points for a smoother curve
    y_smooth = make_interp_spline(bin_centers, n)(x_smooth)

    # Plot the smoothed curve
    # plt.plot(x_smooth, y_smooth, 'g-', linewidth=2)
    
    plt.fill_between(x_smooth, 0, y_smooth, alpha=0.3, color='green')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Probability Density', fontsize=14)  
    ax.set_xlabel('Cosine Similarity', fontsize=14) 
    # Create custom legend entries with colored boxes
    legend_elements = [
        Line2D([0], [0], color='blue', lw=10, alpha=0.3, label='Poison'),
        Line2D([0], [0], color='green', lw=10, alpha=0.3, label='Clean'),
    ]

    # Add legend with custom entries
    ax.legend(handles=legend_elements, handlelength=4, handleheight=3, fontsize=12)

    plt.tight_layout()

    if file_path[-2:] == 'pt':
        plt.savefig('post_pretraining_analysis/clean_poison_dist_%s_%s.png' \
                %(re.search(r"/([^/]+).pt", file_path).group(1), poison_category))
    else:
        plt.savefig('post_pretraining_analysis/clean_poison_dist_%s_%s.png' \
            %(re.search(r"/([^/]+).tsv", file_path).group(1), poison_category))
    plt.close()

plot_clean_n_poison_similarity_distribution('indices/clip_poison_num_compare_update1.tsv', poison_category='less')