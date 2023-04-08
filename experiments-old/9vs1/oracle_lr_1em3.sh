# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=1

# benchmark settings
DATE=Feb15
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

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.001 # learning rate

OLD_VS_NEW="9v1" ## BE MINDFUL OF THIS PARAM


LOSS_TYPES=("base")

# INTITIAL EXPERIMENTS

for ((i=0;i<${#LOSS_TYPES[@]};++i))
do
        LOSS_TYPE=(${LOSS_TYPES[i]})

        OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/loss_${LOSS_TYPE}
        # make save directory
        mkdir -p $OUTDIR

        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT \
                --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE  --batch_size $BS --loss_type $LOSS_TYPE \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type resnet \
                --learner_type default --learner_name NormalNN --oracle_flag \
                --log_dir ${OUTDIR}

done