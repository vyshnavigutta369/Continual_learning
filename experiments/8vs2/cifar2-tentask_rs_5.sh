# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=0

# benchmark settings
DATE=Jan25
DATASET=CIFAR2
FIRST_SPLIT=8
OTHER_SPLIT=2

###############################################################

# load saved models
OVERWRITE=1

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="100" # epochs
MODELNAME=resnet18 
BS=128 # batch size
RS=5 # replay size per class
WD=0.0002 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.01 # learning rate

OLD_VS_NEW="8v2" ## BE MINDFUL OF THIS PARAM

OUTDIR=_outputs/${DATE}/twotask_${DATASET}_replay_${RS}/${OLD_VS_NEW}

# make save directory
mkdir -p $OUTDIR


# INTITIAL EXPERIMENTS


python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS --batch_size_replay $RS\
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --overwrite $OVERWRITE --max_task $MAXTASK \
        --model_name $MODELNAME --model_type resnet \
        --learner_type default --learner_name NormalNN \
        --log_dir ${OUTDIR}

# if [ $EXP_FLAG -eq -1 ] 
# then
    
#     # Oracle - train on every task with full memory
#     # e.g., task 1, task 1+2, ...
#     python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --overwrite $OVERWRITE --max_task $MAXTASK \
#         --model_name $MODELNAME --model_type resnet \
#         --learner_type default --learner_name NormalNN --oracle_flag \
#         --log_dir ${OUTDIR}/oracle

#     # UB - joint training upper bound
#     python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $UB_SPLIT_SIZE --other_split_size 0 --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --overwrite $OVERWRITE --max_task $MAXTASK \
#         --model_name $MODELNAME --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --log_dir ${OUTDIR}/UB

#     # naive fine-tuning - a continual learning agent that trains for full epochs on each task with high forgetting
#     python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#         --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS \
#         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#         --overwrite $OVERWRITE --max_task $MAXTASK \
#         --model_name $MODELNAME --model_type resnet \
#         --learner_type default --learner_name NormalNN \
#         --log_dir ${OUTDIR}/base
# fi

# for UB_RATIO in 0.98 0.95 0.99 1.0
# do
#     # naming convention - loss function + replay sampling strategy
#     if [ $EXP_FLAG -eq 0 ] 
#     then
#         # Base - Random Sample
#         python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS --batch_size_replay $BS \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --overwrite $OVERWRITE --max_task $MAXTASK \
#             --model_name $MODELNAME --model_type resnet \
#             --learner_type rehearsal --learner_name Replay \
#             --loss_type base --replay_type random_sample --oracle_dir ${OUTDIR}/oracle --ub_rat $UB_RATIO \
#             --log_dir ${OUTDIR}/${UB_RATIO}/base_random-sample

#     fi
#     if [ $EXP_FLAG -eq 1 ] 
#     then

#         # Base - Gradient Weighted Class Balancing
#         python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS --batch_size_replay $BS \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --overwrite $OVERWRITE --max_task $MAXTASK \
#             --model_name $MODELNAME --model_type resnet \
#             --learner_type rehearsal --learner_name Replay \
#             --loss_type base --replay_type gradient_cb --oracle_dir ${OUTDIR}/oracle --ub_rat $UB_RATIO \
#             --log_dir ${OUTDIR}/${UB_RATIO}/base_gradient-cb

#     fi
#     if [ $EXP_FLAG -eq 2 ] 
#     then

#         # PredKD - Random Sample
#         python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS --batch_size_replay $BS \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --overwrite $OVERWRITE --max_task $MAXTASK \
#             --model_name $MODELNAME --model_type resnet \
#             --learner_type rehearsal --learner_name Replay \
#             --loss_type pred_kd --replay_type random_sample --oracle_dir ${OUTDIR}/oracle --ub_rat $UB_RATIO \
#             --log_dir ${OUTDIR}/${UB_RATIO}/pred-kd_random-sample

#     fi
#     if [ $EXP_FLAG -eq 3 ] 
#     then

#         # PredKD - Gradient Weighted Class Balancing
#         python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
#             --first_split_size $SPLIT --other_split_size $SPLIT --schedule $SCHEDULE --schedule_type decay --batch_size $BS --batch_size_replay $BS \
#             --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
#             --overwrite $OVERWRITE --max_task $MAXTASK \
#             --model_name $MODELNAME --model_type resnet \
#             --learner_type rehearsal --learner_name Replay \
#             --loss_type pred_kd --replay_type gradient_cb --oracle_dir ${OUTDIR}/oracle --ub_rat $UB_RATIO \
#             --log_dir ${OUTDIR}/${UB_RATIO}/pred-kd_gradient-cb

#     fi
# done    