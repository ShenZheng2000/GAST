# GAST Dataset Setup

Data lives on node-local `/scratch` (per node: run training on the node that holds it).

## 1. Make folders
```bash
mkdir -p /scratch/shenzhen/Datasets/Occ3D_nuscenes /scratch/shenzhen/Datasets/infos
```

## 2. Download
- Occ3D-nuScenes: https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view → unzip into `/scratch/shenzhen/Datasets/Occ3D_nuscenes/` (gives `gts/` and `annotations.json`)
- Info pkls: https://drive.google.com/drive/folders/1nXMyP9FueCuJ-1kFLvFbzvdnaUkwL1L6 → put into `/scratch/shenzhen/Datasets/infos/`
```bash
cd /scratch/shenzhen/Datasets/infos
gdown 1WwxxYji-z5Uy8sGTVMoJMLPlGtPz-IQK -O world-nuscenes_infos_train.pkl
gdown 1y1FINDQ-ZwACiLVaZY-XNAh_w_v1t3dM -O world-nuscenes_infos_val.pkl
```

## 3. Expected layout
```
/scratch/shenzhen/Datasets/
├── Occ3D_nuscenes/
│   ├── gts/scene-xxxx/<token>/labels.npz
│   └── annotations.json
└── infos/
    ├── world-nuscenes_infos_train.pkl
    └── world-nuscenes_infos_val.pkl
```

## 4. Edit `config/gast_nuscenes.yaml`
```yaml
save_params:
  save_path: "/data3/shenzhen/World_Projects/GAST/experiments/Occ3D_nuScenes/"
  save_file:   # replace /mnt/home/chensitao/code/GAST → /data3/shenzhen/World_Projects/GAST
dataset_params:
  label_path: "/scratch/shenzhen/Datasets/Occ3D_nuscenes/"   # keep trailing /
  anno_path:  "/scratch/shenzhen/Datasets/Occ3D_nuscenes/annotations.json"
  train_data_loader:
    pkl_file: "/scratch/shenzhen/Datasets/infos/world-nuscenes_infos_train.pkl"
  val_data_loader:
    pkl_file: "/scratch/shenzhen/Datasets/infos/world-nuscenes_infos_val.pkl"
```

## 5. Edit `tasks/occ3d_nuscenes/run.sh`
Replace `/mnt/home/chensitao/code/GAST` → `/data3/shenzhen/World_Projects/GAST`, then:
```bash
mkdir -p /data3/shenzhen/World_Projects/GAST/logs
cd tasks/occ3d_nuscenes && ./run.sh
```

## 6. Check
```bash
python -c "import pickle;print(pickle.load(open('/scratch/shenzhen/Datasets/infos/world-nuscenes_infos_val.pkl','rb'))['infos'][0]['occ_path'])"
```
Should start with `./data/nuscenes/gts/...` (the dataloader swaps `./data/nuscenes/` for `label_path`). If it doesn't, adjust `label_path` to match.
