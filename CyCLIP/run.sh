#!/bin/bash

runNames='trial_1M_aug_lr'
lpName='trial_1M_aug_lr_eval'
device=1

beginEpoch=21
endEpoch=21
batch_size=256

# clean similarity args
cleanDataPath='../short_100K_clean.csv'
validationPath='../valid_temp.csv'

# LP args
eval_data_types=('CIFAR10' 'CIFAR100') # 'caltech101' 'flowers_102' 'food_101')
# eval_data_types='CIFAR10'

for runName in "${runNames[@]}"
do 
    for eval_data_type in "${eval_data_types[@]}"
    do
    mkdir "logs/$runName/LP_output_logs/"
    mkdir "logs/$runName/ZS_output_logs/"
    mkdir "logs/$runName/LP_output_logs/$eval_data_type"
    mkdir "logs/$runName/ZS_output_logs/$eval_data_type"
    done
done

# poison eval args
dataset='imagenet100'
poison_path='../1M_100_info.csv'

for runName in "${runNames[@]}"
do
    for ((i=$beginEpoch; i<=$endEpoch; i++))
    do
        checkpointPath="logs/$runName/checkpoints/epoch_$i.pt"
        
        # get clean similarity
        # python -m src.main --name $runName --train_data $cleanDataPath --validation_data $validationPath --image_key path --caption_key caption --device_id $device --batch_size $batch_size --epoch $i --checkpoint $checkpointPath  --representation 
        # wait
        
        for eval_data_type in "${eval_data_types[@]}"
        do
            eval_train_data_dir="data/$eval_data_type/train"
            eval_test_data_dir="data/$eval_data_type/test"

            # get LP accuracy
            # python -m src.main --name $lpName --eval_data_type $eval_data_type --eval_train_data_dir $eval_train_data_dir --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath --linear_probe --linear_probe_batch_size $batch_size
            # wait
            # cp "logs/$lpName/output.log" "logs/$runName/LP_output_logs/$eval_data_type/output_epoch$i.log" 
            # wait

            # get ZS accuracy
            python -m src.main --name $lpName --eval_data_type $eval_data_type  --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath 
            wait
            cp "logs/$lpName/output.log" "logs/$runName/ZS_output_logs/$eval_data_type/output_epoch$i.log" 
            wait
        done 

        # get poison evals 
        python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path
        wait
    done
done