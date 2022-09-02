Code for submission WACV 2023 ID 775.

# 1) Datasets:

Please set datasets as follow:

## Prid2011:

prid2011

----> prid_2011

----> splits_prid2011.json

----> splits_single_shot.json

--------> multi_shot

--------> single_shot

## iLIDS-Vid

ilids-vid

----> i-LIDS-VID

--------> i-LIDS-VID

------------> images

------------> sequences

--------> train-test people splits

------------> train_test_splits_ilidsvid.mat

------------> train_test_splits_prid.mat

--------> splits.json

## MARS

mars

----> bbox_test

----> bbox_train

----> info

# 2) Pre-raining the Source

Example for iLIDS-VID

python train_baseline.py --root $PATH_TO_REID_DATA --d ilids-vid --save-dir $SAVE_BASELINE

# 3) Performing UDA:

Example for prid2011 (Source) to ilids-vid (Target)

python train.py --root $PATH_TO_REID_DATA -dt ilids-vid -d prid2011 --train_batch 10 --max_epoch 150 --loss_camcla clc --loss_contr contr --domains source_target --multihead all_head --num_clips 2 --gpu_devices=1 --experiment_name test_delete --resume $SAVE_BASELINE

# 775
