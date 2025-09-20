CFG=$1
DATASET=$1
OUTPUT=$2

SEED=3  # 随机种子
RANK=16 # LoRA秩（模型微调参数）
CTX=4 # 上下文token数量


python train.py --config-file configs/${CFG}.yaml \
--output-dir ${OUTPUT} \
--seed ${SEED}

python inference.py --config-file configs/${CFG}.yaml \
--output-dir ${OUTPUT} \
--seed ${SEED}

python evaluation/eval.py \
--gt_path data/${DATASET}/test/masks \
--seg_path ${OUTPUT}/${DATASET}/seg_results/seed${SEED}/tumor/LORA${RANK}_SHOTS-1_NCTX${CTX}_CSCFalse_CTPend \
--save_path test.csv 