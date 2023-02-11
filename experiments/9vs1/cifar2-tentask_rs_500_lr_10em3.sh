# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=1

# benchmark settings
DATE=Jan25
DATASET=CIFAR2
FIRST_SPLIT=9
OTHER_SPLIT=1

###############################################################

# load saved models
OVERWRITE=1

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="250" # epochs
SCHEDULE_TYPE=cosine
MODELNAME=resnet18 
BS=128 # batch size
RS=500 # replay size per class
WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.001 # learning rate

OLD_VS_NEW="9v1" ## BE MINDFUL OF THIS PARAM

ORACLE_DIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle
OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}
PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/

# make save directory
mkdir -p $OUTDIR


# INTITIAL EXPERIMENTS
python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
        --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
        --overwrite $OVERWRITE --max_task $MAXTASK \
        --model_name $MODELNAME --model_type resnet \
        --learner_type er --learner_name TR \
        --log_dir ${OUTDIR} --plot_dir $PLOT_DIR --oracle_dir $ORACLE_DIR