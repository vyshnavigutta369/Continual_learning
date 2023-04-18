# gpu's to use - can do 1 per experiment for cifar
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}

# benchmark settings
DATE=April-13
DATASET=CIFAR100
FIRST_SPLIT=95
OTHER_SPLIT=5

###############################################################

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="10000" # epochs
SCHEDULE_TYPE=cosine
MODELNAME=vit_pt_imnet 
BS=32 # batch size
WD=0.0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.00005 # learning rate
OLD_VS_NEW="95v5" ## BE MINDFUL OF THIS PARAM

OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/Oracle

# make save directory
mkdir -p $OUTDIR

if [ $GPUID -eq 0 ] 
then  

        # INTITIAL EXPERIMENTS
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE  --batch_size $BS \
                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --oracle_flag \
                --log_dir ${OUTDIR}
fi