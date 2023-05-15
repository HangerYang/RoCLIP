import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

from pathlib import Path

data_types = ['CIFAR10', 'CIFAR100', 'ImageNet1K']
filetemp_poison = "template/%s_imagenet20_%d_0.npz"
filetemp_clean = "result/similarity/%s_%d.npz"
filetemp_lp = "logs/%s/LP_output_logs/%s/output_epoch%d.log"
filetemp_zs = "logs/%s/ZS_output_logs/%s/output_epoch%d.log"

exps = [
        # '1M_per_2_0.5_0.5', 
        # '1M_per_2_1_0.5',
        # '1M_per_2_0.25_0.5',
        # '1M_per_2_0.5_0.5_all_epoch',
        # '1M_per_2_0.9_0.1_all_epoch',
        # '100K_per_3_epoch_random',
        # '100K_clip_no_aug_pb',
        # '100K_clip_pb',
        '100K_clip_pb_20'
        ]

begin_epoch = 1
end_epoch = 32

for exp in exps:
    print('----------%s-----------' % exp)

    clean_sim = []
    top1_poison_rate = []
    top3_poison_rate = []
    top5_poison_rate = []
    
    lp_accuracy = defaultdict(list)
    top1_zs_accuracy = defaultdict(list)
    top3_zs_accuracy = defaultdict(list)
    top5_zs_accuracy = defaultdict(list)
    for i in range(begin_epoch, end_epoch+1):
        print('epoch%d: ' % i)

        data = np.load(filetemp_poison%(exp, i))
        lst = data.files
        for item in lst:
            print(data[item])
        
        top1_poison_rate.append(data['arr_0'])
        top3_poison_rate.append(data['arr_1'])
        top5_poison_rate.append(data['arr_2'])

        # data = np.load(filetemp_clean%(exp, i))
        # print('clean similarity:', data['arr_0'].mean())
        # clean_sim.append(data['arr_0'].mean())
        
        # for data_type in ['CIFAR10', 'CIFAR100', 'ImageNet1K']:
        #     with open(filetemp_lp%(exp, data_type, i), 'r') as f:
        #         # Read the file contents into a string
        #         data = f.read()

        #         # Use a regular expression to search for "accuracy: x.xx"
        #         match = re.search('linear_probe_accuracy:\s+(\d+\.\d+)', data)

        #         if match:
        #             accuracy_str = match.group(1)
        #             accuracy = float(accuracy_str)
        #             lp_accuracy[data_type].append(accuracy)
        #         else:
        #             print("Accuracy not found in file %s epoch %d."%(data_type, i))
        #             exit()

            # with open(filetemp_zs%(exp, data_type, i), 'r') as f:
            #     # Read the file contents into a string
            #     data = f.read()

            #     # Use a regular expression to search for "accuracy: x.xx"
            #     match = re.search('zeroshot_top1:\s+(\d+\.\d+)', data)
            #     match3 = re.search('zeroshot_top3:\s+(\d+\.\d+)', data)
            #     match5 = re.search('zeroshot_top5:\s+(\d+\.\d+)', data)
                
            #     top1_zs_accuracy[data_type].append(float(match.group(1)))
            #     top3_zs_accuracy[data_type].append(float(match3.group(1)))
            #     top5_zs_accuracy[data_type].append(float(match5.group(1)))

    print('----------%s-----------' % exp)
    
    # plt.figure()
    # plt.plot(clean_sim)
    # plt.savefig('result/similarity/%s.png'%exp)
    
    plt.figure()
    plt.plot(top1_poison_rate, label='top1')
    plt.plot(top3_poison_rate, label='top3')
    plt.plot(top5_poison_rate, label='top5')
    
    plt.legend()
    plt.savefig('result/poison_rates/%s.png'%exp)

    # for data_type in data_types:
    #     Path('result/lp_accuracy/%s'%data_type).mkdir(parents=True, exist_ok=True)
    #     Path('result/zs_accuracy/%s'%data_type).mkdir(parents=True, exist_ok=True)
        
    #     plt.figure()
    #     plt.plot(lp_accuracy[data_type])
    #     plt.savefig('result/lp_accuracy/%s/%s.png'%(data_type, exp))
        
    #     plt.figure()
    #     plt.plot(top1_zs_accuracy[data_type], label='top1')
    #     plt.plot(top3_zs_accuracy[data_type], label='top3')
    #     plt.plot(top5_zs_accuracy[data_type], label='top5')
        
    #     plt.legend()
    #     plt.savefig('result/zs_accuracy/%s/%s.png'%(data_type, exp))