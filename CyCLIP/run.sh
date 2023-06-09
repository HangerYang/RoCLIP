#!/bin/bash

runName='10K_inmodal_check'
lpName='1M_every_poison_baseline_eval'
device=4

beginEpoch=1
endEpoch=20
batch_size=256

# clean similarity args
cleanDataPath='../short_100K_clean.csv'
validationPath='../valid_temp.csv'

# LP args
# eval_data_types=('CIFAR10' 'CIFAR100' 'ImageNet1K' 'caltech101' 'flowers_102' 'food_101')
eval_data_types='CIFAR10'

for eval_data_type in "${eval_data_types[@]}"
do
mkdir "logs/$runName/LP_output_logs/"
mkdir "logs/$runName/ZS_output_logs/"
mkdir "logs/$runName/LP_output_logs/$eval_data_type"
mkdir "logs/$runName/ZS_output_logs/$eval_data_type"
done

# poison eval args
dataset='cifarten5'
poison_path='../10K_random_poison_1_5_info.csv'


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
        # python -m src.main --name $lpName --eval_data_type $eval_data_type  --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath 
        # wait
        # cp "logs/$lpName/output.log" "logs/$runName/ZS_output_logs/$eval_data_type/output_epoch$i.log" 
        # wait
    done 

    # get poison evals 
    python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path
    wait
done