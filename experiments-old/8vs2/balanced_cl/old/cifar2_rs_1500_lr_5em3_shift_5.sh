# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=0

# benchmark settings
DATE=Mar13
DATASET=CIFAR2
FIRST_SPLIT=8
OTHER_SPLIT=2

###############################################################

# load saved models
OVERWRITE=0

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="100" # epochs
SCHEDULE_TYPE=cosine
MODELNAME=resnet18 
BS=128 # batch size
RS=1500

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.005 # learning rate

OLD_VS_NEW="8v2" ## BE MINDFUL OF THIS PARAM


# REPLAY_TYPES=("random_sample" "gradient_cb")
REPLAY_TYPES=("random_sample")
LOSS_TYPES=("base")

DEBUG=0
if [ $DEBUG -eq 1 ] 
then   
    TEMP=_temp
    SCHEDULE="10"
    OVERWRITE=1
else
    TEMP=''
fi

WEIGHT_WITH=(5)


for ((i=0;i<${#LOSS_TYPES[@]};++i))
do
        LOSS_TYPE=(${LOSS_TYPES[i]})
        # echo $LOSS_TYPE
        for ((j=0;j<${#REPLAY_TYPES[@]};++j))
        do
                REPLAY_TYPE=(${REPLAY_TYPES[j]})
                # echo $REPLAY_TYPE
                for ((k=0;k<${#WEIGHT_WITH[@]};++k))  ## weight with dist shift
                do                
                        SHIFT=(${WEIGHT_WITH[k]})
                
                        ORACLE_DIR=_outputs/Mar2/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/bh_reptype_${REPLAY_TYPE}_loss_base
                        OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/bh_reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_shift_${SHIFT}${TEMP}
                        PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/bh_replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_shift_${SHIFT}${TEMP}/

                        # make save directory
                        mkdir -p $OUTDIR

                        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE --weight_with $SHIFT \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --overwrite $OVERWRITE --max_task $MAXTASK \
                                --model_name $MODELNAME --model_type resnet \
                                --learner_type er --learner_name TR \
                                --log_dir ${OUTDIR} --plot_dir $PLOT_DIR --oracle_dir $ORACLE_DIR --with_class_balance 1

                
                done
        done
done