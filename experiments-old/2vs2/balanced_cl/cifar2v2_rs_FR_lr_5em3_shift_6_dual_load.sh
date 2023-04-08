# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=1

# benchmark settings
DATE=Mar16
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

DEBUG=0
if [ $DEBUG -eq 1 ] 
then   
    TEMP=_temp
    SCHEDULE="10"
    OVERWRITE=1
else
    TEMP=''
fi

WEIGHT_WITH=(6)
# self.class_replay_ratios = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1} # 84.39
# self.class_replay_ratios = { 0: 2, 1: 1, 2: 3, 3: 3, 4: 2, 5: 2, 6: 2, 7: 1}  # 84.05
# self.class_replay_ratios = { 0: 3, 1: 1, 2: 5, 3: 5, 4: 3, 5: 3, 6: 3, 7: 1}  # 84.45
# self.class_replay_ratios = { 0: 4, 1: 1, 2: 7, 3: 7, 4: 4, 5: 4, 6: 4, 7: 1}  # 84.14
clratios1='{"airplane":1,"automobile":1,"deer":2,"dog":2,"frog":2,"horse":1,"ship":1,"truck":1,"bird":2,"cat":2}'
clratios2='{"airplane":2,"automobile":1,"deer":3,"dog":3,"frog":3,"horse":2,"ship":2,"truck":1,"bird":3,"cat":3}'
clratios3='{"airplane":3,"automobile":1,"deer":5,"dog":5,"frog":5,"horse":3,"ship":3,"truck":1,"bird":5,"cat":5}'
clratios4='{"airplane":4,"automobile":1,"deer":7,"dog":7,"frog":7,"horse":4,"ship":4,"truck":1,"bird":7,"cat":7}'
clratios5='{"airplane":5,"automobile":1,"deer":10,"dog":10,"frog":10,"horse":5,"ship":5,"truck":1,"bird":10,"cat":10}'

clratios_arr=($clratios1 $clratios2 $clratios3 $clratios4 $clratios5)
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
                        for ((l=0;l<${#clratios_arr[@]};++l))  ## weight with dist shift
                        # for ((l=4;l>=2;--l))  ## weight with dist shift
                        do    
                                SHIFT=(${WEIGHT_WITH[k]})
                                DUAL_DATALOADER=True
                                clratios=(${clratios_arr[l]})
                        
                                ORACLE_DIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}
                                OUTDIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}_shift_${SHIFT}_ratio_$l/bh_reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_dual_load${TEMP}
                                
                                # make save directory
                                mkdir -p $OUTDIR

                                python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                        --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                        --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE --weight_with $SHIFT --class_ratios $clratios \
                                        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                        --overwrite $OVERWRITE --max_task $MAXTASK \
                                        --model_name $MODELNAME --model_type resnet \
                                        --learner_type er --learner_name TR \
                                        --log_dir ${OUTDIR} --oracle_dir $ORACLE_DIR --with_class_balance 1 --dual_dataloader $DUAL_DATALOADER  
                        done

                
                done
        done
done