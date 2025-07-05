
source scripts/setup/setup.sh
MODEL=${1:-"NousResearch/Meta-Llama-3-8B-Instruct"}

CONFIGS=(
    "llama3-8b-glue.yaml"

)


CHECKPOINT_PATH=checkpoints/$MODEL/GLUE

TASKS=(
    rte
    cola
    stsb
    sst2
    qnli
    mrpc
    mnli
    qqp
    wnli
)

SEEDS=(42 43 44 45 46)

for TASK_NAME in ${TASKS[@]}
do

    for CONFIG in ${CONFIGS[@]}
    do
        #* Check SVD Estimator:
        bash scripts/setup/svd_estimator.sh $MODEL configs/sparsity/$CONFIG

        for SEED in ${SEEDS[@]}
        do

            if [ "$TASK_NAME" == "mrpc" ] || [ "$TASK_NAME" == "wnli" ]; then
                num_train_epochs=5
            else
                num_train_epochs=3
            fi


            torchrun --nproc_per_node=8 \
                spft/train/run_glue.py \
                --report_to wandb \
                --output_dir $CHECKPOINT_PATH \
                --seed $SEED \
                --model_name_or_path $MODEL \
                --model_short_name $MODEL_NAME \
                --task_name $TASK_NAME \
                --do_train \
                --do_eval \
                --peft lora \
                --config configs/train/glue_train.yaml \
                --spft configs/sparsity/$CONFIG \
                --num-train-epochs $num_train_epochs \
                "$@"  # <<-- forward all extra args

        done
    done
done
