#bash


python main_flow_matching.py \
    --target_dataset "gaussian" \
    --source_dataset "gaussian" \
    --image_size 32\
    --save_dir ./results \
    --epochs 100 \
    --batch_size 64 \
    --steps_sampler 40 \
    --arch "unet_tiny" \
    --schedule "linear" \
    --sample_every 10 \
    --samples_n 4\
    --lr 3e-4\
    --ckpt_every 40

# python main_flow_matching.py \
#     --target_dataset "./data/disks" \
#     --source_dataset "./data/disks" \
#     --save_dir ./results \
#     --epochs 100 \
#     --batch_size 64 \
#     --steps_sampler 40 \
#     --arch "unet_tiny" \
#     --schedule "linear" \
#     --sample_every 10 \
#     --samples_n 4\
#     --lr 3e-4\
#     --ckpt_every 40

# python main_flow_matching.py \
#    --target_dataset "mnist" \
#    --source_dataset "gaussian" \
#    --save_dir ./results \
#    --epochs 100 \
#    --batch_size 64 \
#    --steps_sampler 40 \
#    --arch "unet_tiny" \
#    --schedule "linear" \
#    --sample_every 1 \
#    --samples_n 4\
#    --lr 3e-4\
#    --ckpt_every 40
`

# python main_flow_matching.py \
#     --target_dataset "mnist" \
#     --source_dataset "gaussian" \
#     --save_dir ./results \
#     --epochs 100 \
#     --batch_size 128 \
#     --steps_sampler 40 \
#     --arch "unet_tiny" \
#     --schedule "linear" \
#     --sample_every 1 \
#     --samples_n 4\
#     --lr 3e-4\
#     --ckpt_every 40

#python main_flow_matching.py \
#    --target_dataset "./data/disks/" \
#    --source_dataset "./data/disks/" \
#    --save_dir ./results \
#    --image_size 64 \
#    --epochs 100 \
#    --batch_size 128 \
#    --steps_sampler 40 \
#    --arch "unet_tiny" \
#    --schedule "linear" \
#    --sample_every 1 \
#    --samples_n 4\
#    --lr 3e-4\
#    --ckpt_every 40