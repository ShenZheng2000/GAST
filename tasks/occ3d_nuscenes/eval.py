import numpy as np
import torch
import time
import yaml
import argparse
import logging
import tqdm
import os
import sys
sys.path.append("../..")

from models.gast import GAST
from dataloader.dataset_nus_occ3d import NuscenesOcc3D, collate_fn_default
from utils.iou_eval import IOUEval


def initLogger(log_dir):
    logger = logging.getLogger("console")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    file_handle = logging.FileHandler(os.path.join(log_dir, "infer_results.log"))
    file_handle.setFormatter(file_formatter)
    console_handle = logging.StreamHandler(sys.stdout)
    console_handle.setFormatter(console_formatter)
    logger.addHandler(file_handle)
    logger.addHandler(console_handle)
    return logger

# get configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="path to config file")
args = parser.parse_args()
config = yaml.safe_load(open(args.config_path, 'r'))
if '_base_' in config:  # inherit from base yaml, override with the keys given here
    def merge(b, o):
        for k, v in o.items():
            b[k] = merge(b.get(k, {}), v) if isinstance(v, dict) else v
        return b
    base = yaml.safe_load(open(config.pop('_base_'), 'r'))
    config = merge(base, config)
save_dir = os.path.join(config['infer_params']['save_dir'], "Evaluation")
os.makedirs(save_dir, exist_ok=True)

# init logger
log_dir = os.path.join(save_dir, "log")
os.makedirs(log_dir, exist_ok=True)
logger = initLogger(log_dir)

# load trained model
model = GAST(config=config)
state_dict = torch.load(config['infer_params']['model_path'], map_location="cpu")
new_weights_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    new_weights_dict[new_k] = v
model.load_state_dict(new_weights_dict)
model.cuda()
model.eval()

# init voxel metrics
ssc_iou_meter_list = []
for _ in range(4):
    ssc_iou_meter_list.append(
        IOUEval(
            n_classes=config['model_params']['num_classes'],
            device=torch.device("cpu"),
            is_distributed=False
        )
    )
sc_iou_meter_list = []
for _ in range(4):
    sc_iou_meter_list.append(
        IOUEval(
            n_classes=2,
            device=torch.device("cpu"),
            is_distributed=False         
        )
    )

# init data_loader
valid_dataset = NuscenesOcc3D(
    config=config, 
    dataset_type="valid"
    )
cls_name = valid_dataset.class_names
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_default
)

pbar = tqdm.tqdm(total=len(valid_loader))

for i, data_dict in enumerate(valid_loader):
    with torch.no_grad():

        ## cpu -> gpu
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()
        start_frame, mid_frame, end_frame = \
            0, data_dict['his_occ'].shape[1], data_dict['his_occ'].shape[1] + data_dict['fut_occ'].shape[1]

        prediction, _, _ = model(data_dict)
        future_pred = prediction[:, mid_frame:end_frame].clone()

        ## future occupancy
        for idx in range(3):
            part_future_pred = future_pred[:, idx*2: (idx+1)*2].clone()
            part_future_label = data_dict['fut_occ'][:, idx*2: (idx+1)*2].clone()

            ssc_label_eval = part_future_label.reshape(-1, 1)
            ssc_logits_eval = part_future_pred.flatten(0, 1).argmax(dim=1).reshape(-1, 1)
            ssc_iou_meter_list[idx].addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
            sc_logits_eval = ssc_logits_eval.lt(17).long()
            sc_label_eval = ssc_label_eval.lt(17).long()
            sc_iou_meter_list[idx].addBatch(sc_logits_eval, sc_label_eval)

        all_future_pred = future_pred.clone()
        all_future_label = data_dict['fut_occ'].clone()

        ssc_label_eval = all_future_label.reshape(-1, 1)
        ssc_logits_eval = all_future_pred.flatten(0, 1).argmax(dim=1).reshape(-1, 1)
        ssc_iou_meter_list[-1].addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
        sc_logits_eval = ssc_logits_eval.lt(17).long()
        sc_label_eval = ssc_label_eval.lt(17).long()
        sc_iou_meter_list[-1].addBatch(sc_logits_eval, sc_label_eval)

        pbar.update(1)


for idx in range(4):
    ssc_iou_meter = ssc_iou_meter_list[idx]
    sc_iou_meter = sc_iou_meter_list[idx]
    _, ssc_class_iou = ssc_iou_meter.getIoU()
    _, ssc_class_acc = ssc_iou_meter.getAcc()
    _, ssc_class_recall = ssc_iou_meter.getRecall()
    _, sc_iou = sc_iou_meter.getIoU()

    if idx == 0:
        log_str_start = "0s ~ 1s: start"
    elif idx == 1:
        log_str_start = "1s ~ 2s: start"
    elif idx == 2:
        log_str_start = "2s ~ 3s: start"
    elif idx == 3:
        log_str_start = "0s ~ 3s: start"
    logger.info(log_str_start)

    log_str = ">>> Valid End: [ssc_miou_1_1:{:0.4f}] [sc_iou_1_1:{:0.4f}]".format(
                    ssc_class_iou[:17].mean().item(), sc_iou[1].item()) 
    logger.info(log_str)

    for k, v in enumerate(cls_name):
        valid_log = "class:{} [SSC_IoU_1_1:{:0.4f}] [SSC_Acc_1_1:{:0.4f}] [SSC_Recall_1_1:{:0.4f}]".format(
                    v, ssc_class_iou[k].item(), ssc_class_acc[k].item(), ssc_class_recall[k].item())
        logger.info(valid_log) 
    
    if idx == 0:
        log_str_end = "0s ~ 1s: end"
    elif idx == 1:
        log_str_end = "1s ~ 2s: end"
    elif idx == 2:
        log_str_end = "2s ~ 3s: end"
    elif idx == 3:
        log_str_end = "0s ~ 3s: end"
    logger.info(log_str_end)

    logger.info("########################################")
    logger.info("########################################")