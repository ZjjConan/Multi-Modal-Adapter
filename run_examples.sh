GPU_ID=$1
SEED=$2

export CUDA_VISIBLE_DEVICES=${GPU_ID}

for DATASET in caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
do
    bash scripts/mmadapter/base2new_train.sh ${DATASET} ${SEED} vit_b16_ep5
    bash scripts/mmadapter/base2new_test.sh ${DATASET} ${SEED} vit_b16_ep5 5
done

for DATASET in imagenet
do
    bash scripts/mmadapter/base2new_train.sh ${DATASET} ${SEED} vit_b16_ep5_imnet
    bash scripts/mmadapter/base2new_test.sh ${DATASET} ${SEED} vit_b16_ep5_imnet 5
done