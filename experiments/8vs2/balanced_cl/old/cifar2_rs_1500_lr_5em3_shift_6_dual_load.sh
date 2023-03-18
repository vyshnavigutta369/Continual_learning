# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=1

# benchmark settings
DATE=Mar13
DATASET=CIFAR10
FIRST_SPLIT=2
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
RS=5000

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.005 # learning rate

OLD_VS_NEW=${FIRST_SPLIT}v${OTHER_SPLIT}


# REPLAY_TYPES=("random_sample" "gradient_cb")
REPLAY_TYPES=("random_sample")
LOSS_TYPES=("base")

DEBUG=1
if [ $DEBUG -eq 1 ] 
then   
    TEMP=_temp
    SCHEDULE="10"
    OVERWRITE=1
else
    TEMP=''
fi

WEIGHT_WITH=(3)
# declare -A clratios=( ["airplane"]="1" ["automobile"]="2" ["deer"]="1" ["dog"]="2" ["frog"]="1" ["horse"]="2"  ["ship"]="1" ["truck"]="2" ["bird"]="1" ["cat"]="2" )
clratios='{"airplane":1,"automobile":1,"deer":1,"dog":1,"frog":1,"horse":1,"ship":1,"truck":1,"bird":1,"cat":1}'
# clratios='{"name":1,"name2":1}'
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
                        DUAL_DATALOADER=True
                
                        ORACLE_DIR=_outputs/Mar2/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/bh_reptype_${REPLAY_TYPE}_loss_base
                        OUTDIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/bh_reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_shift_${SHIFT}${TEMP}_dual_load
                        
                        # make save directory
                        mkdir -p $OUTDIR

                        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE --weight_with $SHIFT --class_ratios $clratios \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --overwrite $OVERWRITE --max_task $MAXTASK \
                                --model_name $MODELNAME --model_type resnet \
                                --learner_type er --learner_name TR \
                                --log_dir ${OUTDIR} --with_class_balance 1 --dual_dataloader $DUAL_DATALOADER

                
                done
        done
done