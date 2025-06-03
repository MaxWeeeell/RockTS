is_train=1
model_type='rockts'
context_points=512
target_points=96
batch_size=64

n_layers=3
n_heads=16

patch_len=48
stride=48

revin=1

n_epochs=20
lr=1e-4

random_seed=2021

for dset in 'ETTh1_anom' 'ETTh2_anom' 'ETTm1_anom' 'ETTm2_anom' 'weather_anom' 'electricity_anom' 'traffic_anom' 'solar_anom'
do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs" ]; then
        mkdir logs
    fi
    if [ ! -d "logs/$model_type" ]; then
        mkdir logs/$model_type
    fi
    if [ ! -d "logs/$model_type/$dset" ]; then
        mkdir logs/$model_type/$dset
    fi

    for target_points in 96 192 336 720
    do
        python -u rockts_train.py \
        --is_train $is_train \
        --dset $dset \
        --context_points $context_points \
        --target_points $target_points \
        --batch_size $batch_size \
        --patch_len $patch_len\
        --stride $stride\
        --revin 1 \
        --n_layers $n_layers\
        --n_heads $n_heads \
        --d_model 128 \
        --d_ff 256\
        --dropout 0.2\
        --head_drop 0 \
        --n_epochs $n_epochs\
        --lr $lr \
        --model_type $model_type\  >logs/$model_type/$dset/'Context'$context_points'_target'$target_points'_patchlen'$patch_len'_result'.log 
    done
done
