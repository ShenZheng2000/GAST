# # train
# CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=59999 --use_env main.py \
# --config_path /data3/shenzhen/World_Projects/GAST/config/gast_nuscenes.yaml \
# 2>&1 | tee /data3/shenzhen/World_Projects/GAST/logs/nus.txt

# eval, GT occ + predicted ego traj (weights/gast_gtocc_predtraj.pth)
CUDA_VISIBLE_DEVICES="0" \
python eval.py --config_path /data3/shenzhen/World_Projects/GAST/config/gast_eval_predtraj.yaml

# # eval, GT occ + GT ego traj (weights/gast_gtocc_gttraj.pth)
# CUDA_VISIBLE_DEVICES="1" \
# python eval.py --config_path /data3/shenzhen/World_Projects/GAST/config/gast_eval_gttraj.yaml
