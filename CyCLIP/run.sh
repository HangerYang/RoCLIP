#!/bin/bash

runName='100K_per_3_epoch_random'
device=7

beginEpoch=16
endEpoch=32
batch_size=256

# clean similarity args
cleanDataPath='../short_100K_clean.csv'
validationPath='../valid_temp.csv'

# LP args
eval_data_types=('CIFAR10' 'CIFAR100' 'ImageNet1K')


for eval_data_type in "${eval_data_types[@]}"
do
mkdir "logs/$runName/LP_output_logs/$eval_data_type"
mkdir "logs/$runName/ZS_output_logs/$eval_data_type"
done

# poison eval args
dataset='imagenet100'
poison_path='../random_poison_80_info.csv'


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
        python -m src.main --name $runName --eval_data_type $eval_data_type --eval_train_data_dir $eval_train_data_dir --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath --linear_probe --linear_probe_batch_size $batch_size
        wait
        cp "logs/$runName/output.log" "logs/$runName/LP_output_logs/$eval_data_type/output_epoch$i.log" 
        wait

        # get ZS accuracy
        python -m src.main --name $runName --eval_data_type $eval_data_type  --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath 
        wait
        cp "logs/$runName/output.log" "logs/$runName/ZS_output_logs/$eval_data_type/output_epoch$i.log" 
        wait
    done 

    # get poison evals 
    # python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path
    # wait
done

