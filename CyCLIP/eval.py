import numpy as np
import matplotlib.pyplot as plt


filetemp_poison = "template/%s_imagenet512_%d_0.npz"
filetemp_clean = "result/similarity/%s_%d.npz"

exps = ['1M_per_2_0.5_0.5', 
        '1M_per_2_1_0.5',
        '1M_per_2_0.25_0.5',
        '1M_per_2_0.5_0.5_all_epoch',]

begin_epoch = 1
end_epoch = 5

for exp in exps:
    print('----------%s-----------' % exp)

    clean_sim = []
    for i in range(begin_epoch, end_epoch+1):
        print('epoch%d: ' % i)

        data = np.load(filetemp_poison%(exp, i))
        lst = data.files
        for item in lst:
            print(data[item])

        data = np.load(filetemp_clean%(exp, i))
        print('clean similarity:', data['arr_0'].mean())
        clean_sim.append(data['arr_0'].mean())
    print('----------%s-----------' % exp)
    
    plt.figure()
    plt.plot(clean_sim)
    plt.savefig('result/similarity/%s.png'%exp)