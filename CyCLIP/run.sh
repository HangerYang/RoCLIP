#!/bin/bash

runName='SafeCLIP_CC3M_backdoors'
lpName='RoCLIP_CC3M_backdoors_continue_eval'
device=5

beginEpoch=56
endEpoch=56
batch_size=256


dataset='imagenet1500'
poison_path='/home/hyang/NNCLIP/1M_500_16_info.csv'
# eval_data_types=('DTD')
# eval_data_types=('DTD' 'OxfordIIITPet' 'CIFAR10' 'CIFAR100' 'FGVCAircraft' 'Flowers102' 'Food101')
#  'DTD' 'OxfordIIITPet' 'StanfordCars' 'ImageNet1K' 'Caltech101'
# for eval_data_type in "${eval_data_types[@]}"
# do
# mkdir "logs/$runName/LP_output_logs/"
# mkdir "logs/$runName/ZS_output_logs/"
# mkdir "logs/$runName/LP_output_logs/$eval_data_type"
# mkdir "logs/$runName/ZS_output_logs/$eval_data_type"
# done

# for runName in "${runNames[@]}"
# do
for ((i=$beginEpoch; i<=$endEpoch; i=i+1))
    do
        checkpointPath="logs/$runName/checkpoints/epoch_$i.pt"
        
        # get clean similarity
        # python -m src.main --name $runName --train_data $cleanDataPath --validation_data $validationPath --image_key path --caption_key caption --device_id $device --batch_size $batch_size --epoch $i --checkpoint $checkpointPath  --representation 
        # wait
        
        for eval_data_type in "${eval_data_types[@]}"
        do
            eval_train_data_dir="data/$eval_data_type/train"
            eval_test_data_dir="data/$eval_data_type/test"

            # get ZS accuracy
        #     python -m src.main --name $lpName --eval_data_type $eval_data_type  --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath 
        #     wait
        #     cp "logs/$lpName/output.log" "logs/$runName/ZS_output_logs/$eval_data_type/output_epoch$i.log" 
        #     wait

        # # #     # get LP accuracy
        #     python -m src.main --name $lpName --eval_data_type $eval_data_type --eval_train_data_dir $eval_train_data_dir --eval_test_data_dir $eval_test_data_dir --device_id $device --checkpoint $checkpointPath --linear_probe --linear_probe_batch_size $batch_size
        #     wait
        #     cp "logs/$lpName/output.log" "logs/$runName/LP_output_logs/$eval_data_type/output_epoch$i.log" 
        #     wait
        # done
        # # get poison evals 
        python verify_with_template_full.py --model_name $runName --device $device --epoch $i --dataset $dataset --path $poison_path --distributed
    done
done


# python -m src.main --name NNCLIP_1M_eval --eval_data_type StanfordCars --eval_test_data_dir data/stanford_cars --device_id 0 --checkpoint /home/hyang/NNCLIP/CyCLIP/logs/NNCLIP_1M/checkpoints/epoch_38.pt