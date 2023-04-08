# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=0

# benchmark settings
DATE=Apr5
DATASET=SUPER-CIFAR100
FIRST_SPLIT=10
OTHER_SPLIT=10

###############################################################

# load saved models
OVERWRITE=1

# number of tasks to run
MAXTASK=-1 # run every task

# hard coded inputs
REPEAT=1
SCHEDULE="100" # epochs
STEPS="2000"

SCHEDULE_TYPE=cosine
MODELNAME=resnet18 
MODELTYPE=resnet
LR=0.005 # learning rate
BS=128 # batch size
RS=128


# SCHEDULE_TYPE=decay
# MODELNAME=vit_pt_imnet
# MODELTYPE=zoo
# LR=0.00001
# BS=16 # batch size
# RS=16

RN=1000 ## replay samples

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer



OLD_VS_NEW=${FIRST_SPLIT}v${OTHER_SPLIT}


# REPLAY_TYPES=("random_sample" "gradient_cb")
REPLAY_TYPES=("random_sample")
# REPLAY_STRATEGIES=('loss' 'logit_dist_proba_shift_min' 'confidence_proba_shift_min' 'margin_proba_shift_min' 'replay_count_proba_shift_min' 'logit_dist_proba_shift')
REPLAY_STRATEGIES=('margin_proba_shift_min')
LOSS_TYPES=("base")
# clratios='{"airplane":3,"automobile":1,"deer":5,"dog":5,"frog":5,"horse":3,"ship":3,"truck":1,"bird":5,"cat":5}'
# clratios='{"airplane":2,"automobile":1,"deer":3,"dog":3,"frog":3,"horse":2,"ship":2,"truck":1,"bird":3,"cat":3}'
DEBUG=0
if [ $DEBUG -eq 1 ] 
then   
    TEMP=_temp
    SCHEDULE="10"
    OVERWRITE=1
else
    TEMP=''
fi

CLASS_WEIGHTING_WITH=(12)

for ((i=0;i<${#LOSS_TYPES[@]};++i))
do
        LOSS_TYPE=(${LOSS_TYPES[i]})
        # echo $LOSS_TYPE
        for ((j=0;j<${#REPLAY_TYPES[@]};++j))
        do
                REPLAY_TYPE=(${REPLAY_TYPES[j]})
                # echo $REPLAY_TYPE
                for ((k=0;k<${#CLASS_WEIGHTING_WITH[@]};++k))  
                do           
                        
                        for ((l=0;l<${#REPLAY_STRATEGIES[@]};++l))  
                        do 

                                SHIFT=(${CLASS_WEIGHTING_WITH[k]})
                                REPLAY_STRATEGY=(${REPLAY_STRATEGIES[l]})
                                DUAL_DATALOADER=True
                                CUSTOM_REPLAY_LOADER=True

                                # ORACLE_DIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/reptype_${REPLAY_TYPE}_loss_base${TEMP}
                                ORACLE_DIR=_outputs/Feb20/twotask_${OLD_VS_NEW}_CIFAR2/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}${TEMP}
                                OUTDIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RN}_shift_${SHIFT}/bh_reptype_${REPLAY_TYPE}_repstr_${REPLAY_STRATEGY}_loss_${LOSS_TYPE}${TEMP}
                                
                                # make save directory
                                mkdir -p $OUTDIR

                                python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                        --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --steps $STEPS --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                        --batch_size_replay $RS --replay_type $REPLAY_TYPE  --num_replay_samples $RN --replay_strategy $REPLAY_STRATEGY \
                                        --loss_type $LOSS_TYPE --class_weighting_with $SHIFT --weight_reverse True \
                                        --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                        --overwrite $OVERWRITE --max_task $MAXTASK \
                                        --model_name $MODELNAME --model_type $MODELTYPE \
                                        --learner_type er --learner_name TR \
                                        --log_dir ${OUTDIR} --with_class_balance 1 --custom_replay_loader $CUSTOM_REPLAY_LOADER

                                # OUTDIR=_outputs/${DATE}/${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}_shift_${SHIFT}/bh_reptype_${REPLAY_TYPE}_repstr_${REPLAY_STRATEGY}_loss_${LOSS_TYPE}_b${TEMP}
                                # BATCH_SAMPLER=True
                                # # make save directory
                                # mkdir -p $OUTDIR

                                # python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                #         --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                #         --batch_size_replay $RS --replay_type $REPLAY_TYPE --replay_strategy $REPLAY_STRATEGY \
                                #         --loss_type $LOSS_TYPE --weight_with $SHIFT \
                                #         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                #         --overwrite $OVERWRITE --max_task $MAXTASK \
                                #         --model_name $MODELNAME --model_type resnet \
                                #         --learner_type er --learner_name TR \
                                #         --log_dir ${OUTDIR}  --oracle_dir $ORACLE_DIR --with_class_balance 1 --weighted_sampler $WEIGHTED_SAMPLER --batch_sampler $BATCH_SAMPLER
                        
                        done
                
                done
        done
done