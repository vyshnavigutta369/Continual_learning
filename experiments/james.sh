# bash experiments/cifar100-tentask.sh n

# gpu's to use - can do 1 per experiment for cifar
DEFAULTGPU=0
GPUID=${1:-$DEFAULTGPU}

# benchmark settings
DATE=March_20
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
SCHEDULE="1000" # epochs
SCHEDULE_TYPE=decay
MODELNAME=vit_pt_imnet 
BS=16 # batch size
WD=0.0 # weight decay
MOM=0.9 # momentum
OPT="SGD" # optimizer
LR=0.00001 # learning rate
OLD_VS_NEW="8v2" ## BE MINDFUL OF THIS PARAM

for RS in 1 10 20 50 100 200 #0 # replay size per class
# for RS in 100
do
    ORACLE_DIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}/Oracle
    OUTDIR=_outputs/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}/replay_${RS}
    PLOT_DIR=plots_and_tables/${DATE}/twotask_${OLD_VS_NEW}_${DATASET}/LR_${LR}/replay_${RS}/

    # make save directory
    mkdir -p $OUTDIR
    LR_USE=$LR   
    if [ $GPUID -eq 0 ]
    then 
        # Base
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --mu -1 \
                --log_dir ${OUTDIR}/base --plot_dir $PLOT_DIR/base --oracle_dir $ORACLE_DIR
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --mu -1 --loss_type pred_kd \
                --log_dir ${OUTDIR}/kd-base --plot_dir $PLOT_DIR/kd-base --oracle_dir $ORACLE_DIR
    fi
    if [ $GPUID -eq 1 ] 
    then
        # CB
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --replay_type gradient_cb --mu -1 \
                --log_dir ${OUTDIR}/cb --plot_dir ${PLOT_DIR}/cb --oracle_dir $ORACLE_DIR
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --replay_type gradient_cb --mu -1 --loss_type pred_kd \
                --log_dir ${OUTDIR}/kd-cb --plot_dir ${PLOT_DIR}/kd-cb --oracle_dir $ORACLE_DIR

    fi 
    if [ $GPUID -eq 2 ] 
    then 
        # Base Smart
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --mu 1 \
                --log_dir ${OUTDIR}/base_smart --plot_dir $PLOT_DIR/base_smart --oracle_dir $ORACLE_DIR
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --mu 1 --loss_type pred_kd \
                --log_dir ${OUTDIR}/kd-base_smart --plot_dir $PLOT_DIR/kd-base_smart --oracle_dir $ORACLE_DIR
    fi
    if [ $GPUID -eq 3 ] 
    then

        # CB Smart
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --replay_type gradient_cb --mu 1 \
                --log_dir ${OUTDIR}/cb_smart --plot_dir ${PLOT_DIR}/cb_smart --oracle_dir $ORACLE_DIR
        python -u run.py --dataset $DATASET --train_aug --rand_split --gpuid $GPUID --repeat $REPEAT \
                --first_split_size $FIRST_SPLIT --other_split_size $OTHER_SPLIT --schedule $SCHEDULE --schedule_type $SCHEDULE_TYPE --batch_size $BS --batch_size_replay $RS\
                --optimizer $OPT --lr $LR_USE --momentum $MOM --weight_decay $WD \
                --overwrite $OVERWRITE --max_task $MAXTASK \
                --model_name $MODELNAME --model_type zoo \
                --learner_type er --learner_name TR --replay_type gradient_cb --mu 1 --loss_type pred_kd \
                --log_dir ${OUTDIR}/kd-cb_smart --plot_dir ${PLOT_DIR}/kd-cb_smart --oracle_dir $ORACLE_DIR

    fi 

        

done