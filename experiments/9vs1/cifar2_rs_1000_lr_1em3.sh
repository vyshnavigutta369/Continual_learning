# bash experiments/cifar100-tentask.sh n

# process inputs
DEFAULTEXP=-1
EXP_FLAG=${1:-$DEFAULTEXP}

# gpu's to use - can do 1 per experiment for cifar
GPUID=2

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
SCHEDULE="200" # epochs
SCHEDULE_TYPE=cosine
MODELNAME=resnet18 
BS=128 # batch size
RS=1000 # replay size per class

WD=0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.001 # learning rate

OLD_VS_NEW="9v1" ## BE MINDFUL OF THIS PARAM


REPLAY_TYPES=("random_sample" "gradient_cb")
LOSS_TYPES=("base")

WEIGHT_WITH_DIST_SHIFT=(2)
WEIGHT_WITH_ACC=(2)

for ((i=0;i<${#LOSS_TYPES[@]};++i))
do
        LOSS_TYPE=(${LOSS_TYPES[i]})
        # echo $LOSS_TYPE
        for ((j=0;j<${#REPLAY_TYPES[@]};++j))
        do
                REPLAY_TYPE=(${REPLAY_TYPES[j]})
                # echo $REPLAY_TYPE
                
                ORACLE_DIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/Oracle/loss_${LOSS_TYPE}/
                OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}/
                PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}/

                # make save directory
                mkdir -p $OUTDIR

                # python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                #         --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                #         --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE \
                #         --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                #         --overwrite $OVERWRITE --max_task $MAXTASK \
                #         --model_name $MODELNAME --model_type resnet \
                #         --learner_type er --learner_name TR \
                #         --log_dir ${OUTDIR} --plot_dir $PLOT_DIR --oracle_dir $ORACLE_DIR
                
                for ((k=0;k<${#WEIGHT_WITH_DIST_SHIFT[@]};++k))  ## weight with dist shift
                do                
                        DIST_SHIFT=(${WEIGHT_WITH_DIST_SHIFT[k]})
                        OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_dist_shift_${DIST_SHIFT}/
                        PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_dist_shift_${DIST_SHIFT}/

                        mkdir -p $OUTDIR
                        
                        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE --weight_with_dist_shift $DIST_SHIFT  \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --overwrite $OVERWRITE --max_task $MAXTASK \
                                --model_name $MODELNAME --model_type resnet \
                                --learner_type er --learner_name TR \
                                --log_dir ${OUTDIR} --plot_dir $PLOT_DIR --oracle_dir $ORACLE_DIR
                done

                for ((k=0;k<${#WEIGHT_WITH_ACC[@]};++k))  ## weight with acc shift
                do                
                        ACC=(${WEIGHT_WITH_ACC[k]})
                        OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_acc_shift_${ACC}/
                        PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}_schedule_${SCHEDULE_TYPE}/replay_${RS}/reptype_${REPLAY_TYPE}_loss_${LOSS_TYPE}_acc_shift_${ACC}/
                        
                        mkdir -p $OUTDIR

                        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS \
                                --batch_size_replay $RS --replay_type $REPLAY_TYPE --loss_type $LOSS_TYPE --weight_with_acc $ACC  \
                                --optimizer $OPT --lr $LR --momentum $MOM --weight_decay $WD \
                                --overwrite $OVERWRITE --max_task $MAXTASK \
                                --model_name $MODELNAME --model_type resnet \
                                --learner_type er --learner_name TR \
                                --log_dir ${OUTDIR} --plot_dir $PLOT_DIR --oracle_dir $ORACLE_DIR
                done
        done
done