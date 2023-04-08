# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID='0'

# benchmark settings
DATE=Apr4
DATASET=CIFAR10
FIRST_SPLIT=8
OTHER_SPLIT=2

###############################################################

# load saved models
OVERWRITE=1

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="10" # epochs
STEPS="1000"

SCHEDULE_TYPE=cosine
MODELNAME=resnet18
MODELTYPE=resnet
BS=128 # batch size
LR=0.005 # learning rate

# SCHEDULE_TYPE=decay
# MODELNAME=vit_pt_imnet
# MODELTYPE=zoo
# LR=0.00001
# BS=16 # batch size

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer


OLD_VS_NEW=${FIRST_SPLIT}v${OTHER_SPLIT}

REPLAY_TYPES=("random_sample")
LOSS_TYPES=("base")

DEBUG=0
if [ $DEBUG -eq 1 ] 
then   
    TEMP=_temp
    SCHEDULE="10"
else
    TEMP=''
fi

# INTITIAL EXPERIMENTS

for ((i=0;i<${#LOSS_TYPES[@]};++i))
do
        LOSS_TYPE=(${LOSS_TYPES[i]})

        for ((j=0;j<${#REPLAY_TYPES[@]};++j))
        do
                REPLAY_TYPE=(${REPLAY_TYPES[j]})

                OUTDIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}
                # PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}/
                # make save directory
                mkdir -p $OUTDIR

                python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                        --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT \
                        --schedule $SCHEDULE --steps $STEPS --schedule_type $SCHEDULE_TYPE  --batch_size $BS --loss_type $LOSS_TYPE --replay_type $REPLAY_TYPE \
                        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                        --overwrite $OVERWRITE --max_task $MAXTASK \
                        --model_name $MODELNAME --model_type $MODELTYPE \
                        --learner_type default --learner_name NormalNN --oracle_flag \
                        --log_dir ${OUTDIR} --with_class_balance 1  --dual_dataloader True 

                # OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/bh_reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}
                # PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/bh_reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}/
                # # make save directory
                # mkdir -p $OUTDIR
                
                # python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                #         --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT \
                #         --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE  --batch_size $BS --loss_type $LOSS_TYPE --replay_type $REPLAY_TYPE \
                #         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                #         --overwrite $OVERWRITE --max_task $MAXTASK \
                #         --model_name $MODELNAME --model_type resnet \
                #         --learner_type default --learner_name NormalNN --oracle_flag \
                #         --log_dir ${OUTDIR}  --plot_dir $PLOT_DIR --with_class_balance 1 
        done

done