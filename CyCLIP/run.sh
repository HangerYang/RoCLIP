#!/bin/bash



runName='100K_clip_pb_20'
device=4

beginEpoch=1
endEpoch=32
batch_size=128

if test -e "logs/$runName/output_train.log"; then
  echo "File exists."
else
  cp "logs/$runName/output.log" "logs/$runName/output_train.log"
fi

# clean similarity args
cleanDataPath='../short_100K_clean.csv'
validationPath='../valid_temp.csv'

# LP args
eval_data_types=('CIFAR10' 'CIFAR100' 'ImageNet1K')

mkdir "logs/$runName/LP_output_logs"
mkdir "logs/$runName/ZS_output_logs"
for eval_data_type in "${eval_data_types[@]}"
do
    mkdir "logs/$runName/LP_output_logs/$eval_data_type"
    mkdir "logs/$runName/ZS_output_logs/$eval_data_type"
done

# poison eval args
dataset='imagenet20'
poison_path='../100K_random_poison_20_info_1.csv'


for ((i=$beginEpoch; i<=$endEpoch; i=i+3))
do
    checkpointPath="logs/$runName/checkpoints/epoch_$i.pt"
    
    # get clean similarity
    # python -m src.main --name $runName --train_data $cleanDataPath --validation_data $validationPath --image_key path --caption_key caption --device_id $device --batch_size $batch_size --epoch $i --checkpoint $checkpointPath  --representation 
    # wait
    
    # for eval_data_type in "${eval_data_types[@]}"
    # do
    #     eval_train_data_dir="data/$eval_data_type/train"
    #     eval_test_data_dir="data/$eval_data_type/test"

    #     # get LP accuracy
    #     python -m src.main --name $runName --eval_data_type $eval_data_type --eval_train_data_dir $eval_train_data_dir --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath --linear_probe --linear_probe_batch_size $batch_size
    #     wait
    #     cp "logs/$runName/output.log" "logs/$runName/LP_output_logs/$eval_data_type/output_epoch$i.log" 
    #     wait

    #     # get ZS accuracy
    #     python -m src.main --name $runName --eval_data_type $eval_data_type  --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath 
    #     wait
    #     cp "logs/$runName/output.log" "logs/$runName/ZS_output_logs/$eval_data_type/output_epoch$i.log" 
    #     wait
    # done 

    # get poison evals 
    python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path
    wait
done

