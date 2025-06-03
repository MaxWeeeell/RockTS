is_train=1
model_type='96_input'
context_points=512
target_points=96
batch_size=64

n_layers=3
n_heads=16

patch_len=48
stride=48

revin=1

dset='ETTh1'

n_epochs=20
lr=1e-4
random_seed=2021
cost_lambda=1

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

python -u rockts_train.py \
--is_train $is_train \
--dset $dset \
--context_points $context_points \
--target_points 96 \
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
--seed $random_seed\
--cost_lambda $cost_lambda \
--r 0.9\
--model_type $model_type\  >logs/$model_type/$dset/'Context'$context_points'_target96_patchlen'$patch_len'_costlambda'$cost_lambda'_lr'$lr'_epoch'$epoch.log 

python -u rockts_train.py \
--is_train $is_train \
--dset $dset \
--context_points $context_points \
--target_points 192 \
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
--seed $random_seed\
--cost_lambda $cost_lambda \
--r 0.9\
--model_type $model_type\  >logs/$model_type/$dset/'Context'$context_points'_target192_patchlen'$patch_len'_costlambda'$cost_lambda'_lr'$lr'_epoch'$epoch.log

python -u rockts_train.py \
--is_train $is_train \
--dset $dset \
--context_points $context_points \
--target_points 336 \
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
--seed $random_seed\
--cost_lambda $cost_lambda \
--r 0.9\
--model_type $model_type\  >logs/$model_type/$dset/'Context'$context_points'_target336_patchlen'$patch_len'_costlambda'$cost_lambda'_lr'$lr'_epoch'$epoch.log

python -u rockts_train.py \
--is_train $is_train \
--dset $dset \
--context_points $context_points \
--target_points 720 \
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
--seed $random_seed\
--cost_lambda $cost_lambda \
--r 0.9\
--model_type $model_type\  >logs/$model_type/$dset/'Context'$context_points'_target720_patchlen'$patch_len'_costlambda'$cost_lambda'_lr'$lr'_epoch'$epoch.log