#!/bin/bash

runName='100K_per_2_image'
lpName='100K_per_2_image_eval'
device=6

beginEpoch=1
endEpoch=5
batch_size=256

# clean similarity args
cleanDataPath='../short_100K_clean.csv'
validationPath='../valid_temp.csv'

# LP args
eval_data_type='CIFAR10'
eval_train_data_dir='data/CIFAR10/train'
eval_test_data_dir='data/CIFAR10/test'

mkdir "logs/$runName/output_logs"

# poison eval args
dataset='imagenet100'
poison_path='../random_poison_80_info.csv'


for ((i=$beginEpoch; i<=$endEpoch; i++))
do
    checkpointPath="logs/$runName/checkpoints/epoch_$i.pt"
    
    # get clean similarity
    # python -m src.main --name $runName --train_data $cleanDataPath --validation_data $validationPath --image_key path --caption_key caption --device_id $device --batch_size $batch_size --epoch $i --checkpoint $checkpointPath  --representation 
    # wait
    
    # get LP accuracy
    python -m src.main --name $lpName --eval_data_type $eval_data_type --eval_train_data_dir $eval_train_data_dir --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath --linear_probe --linear_probe_batch_size $batch_size
    
    wait
    
    cp "logs/$lpName/output.log" "logs/$runName/output_logs/output_epoch$i.log" 
    wait

    # get poison evals 
    # python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path
    # wait
done

