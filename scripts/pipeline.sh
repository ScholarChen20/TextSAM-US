CFG=$1  #数据集配置文件 例如:BUSI
DATASET=$1  #数据集标识 例如:BUSI
OUTPUT=$2  # 保存地址 ./output


SEED=3  # 随机种子
RANK=16 # LoRA秩（模型微调参数）
CTX=4 # 上下文token数量

echo "*****begin training*****"
python train.py --config-file configs/${CFG}.yaml \
--output-dir ${OUTPUT} \
--seed ${SEED}

echo "*****begin inference*****"
#python inference.py --config-file configs/${CFG}.yaml \
#--output-dir ${OUTPUT} \
#--seed ${SEED}

echo "*****begin evaluation*****"
#python evaluation/eval.py \
#--gt_path data/${DATASET}/test/masks \
#--seg_path ${OUTPUT}/${DATASET}/seg_results/seed${SEED}/tumor/LORA${RANK}_SHOTS-1_NCTX${CTX}_CSCFalse_CTPend \
#--save_path test.csv