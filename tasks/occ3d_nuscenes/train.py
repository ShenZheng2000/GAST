import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("../..")

from option import Option
from models.gast import GAST
from dataloader.dataset_nus_occ3d import NuscenesOcc3D, collate_fn_default
from utils.warmupLR import WarmupCosineLR
from utils.iou_eval import IOUEval
from utils.avgmeter import AverageMeter
from utils.remain_time import RemainTime
from utils.lovasz_softmax import Lovasz_softmax, lovasz_softmax
from utils.ssc_loss import CE_ssc_loss, geo_scal_loss, sem_scal_loss
from utils.traj_loss import TransLoss, RotationLoss


class Trainer():
    def __init__(self, settings: Option, recorder=None):
        self.settings = settings
        self.config = settings.config
        self.recorder = recorder
        self.model = GAST(config=self.config).cuda()

        # init data_loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # init criterion
        self.criterion_ssc = self._initCriterion_ssc()
        self.criterion_traj = self._initCriterion_traj()

        # init voxel_class_weights
        cls_freq = np.array(self.config['dataset_params']['cls_freq'])
        cls_freq = cls_freq / cls_freq.sum()
        class_weights = np.log(1 + 1 / (cls_freq + 1e-8))
        class_weights = class_weights / class_weights.max()
        self.class_weights = torch.from_numpy(class_weights).float().cuda()

        # init optimizer
        self.optimizer = self._initOptimizer()

        # init lr_scheduler
        self.scheduler = WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.config['train_params']["learning_rate"],
            warmup_steps=self.config['train_params']['warmup_epochs'] * len(self.train_loader),
            momentum=self.config['train_params']["lr_momentum"],
            max_steps=len(self.train_loader) * (self.config['train_params']['max_num_epochs'] - self.config['train_params']['warmup_epochs'])
            )

        # multi_gpu
        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            else:
                self.model = nn.DataParallel(self.model)
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()

        # init voxel metrics
        self.ssc_iou_meter_history = IOUEval(
            n_classes=self.config['model_params']['num_classes'],
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.ssc_iou_meter_history.reset()

        self.sc_iou_meter_history = IOUEval(
            n_classes=2,
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.sc_iou_meter_history.reset()

        self.ssc_iou_meter_future = IOUEval(
            n_classes=self.config['model_params']['num_classes'],
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.ssc_iou_meter_future.reset()

        self.sc_iou_meter_future = IOUEval(
            n_classes=2,
            device=torch.device("cpu"),
            is_distributed=self.settings.distributed            
        )
        self.sc_iou_meter_future.reset()

        # init remain_timer
        self.data_time = AverageMeter()
        self.batch_time_train = AverageMeter()
        self.batch_time_valid = AverageMeter()
        self.remain_time = RemainTime(
            max_epochs=self.config['train_params']['max_num_epochs'],
            steps_per_epoch=len(self.train_loader),
            steps_per_epoch_valid=len(self.val_loader),
            data_time=self.data_time,
            batch_time_train=self.batch_time_train,
            batch_time_valid=self.batch_time_valid
            )


    def _initDataloader(self):
        train_dataset = NuscenesOcc3D(
            config=self.config, 
            dataset_type="train"
            )
        valid_dataset = NuscenesOcc3D(
            config=self.config, 
            dataset_type="valid"
            )

        self.cls_name = train_dataset.class_names

        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=train_dataset,
                shuffle=True,
                drop_last=True
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset=valid_dataset,
                shuffle=False,
                drop_last=False
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.config['dataset_params']['train_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['train_data_loader']['num_workers'],
                drop_last=True,
                collate_fn=collate_fn_default,
                sampler=train_sampler
            )
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=self.config['dataset_params']['val_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['val_data_loader']['num_workers'],
                drop_last=False,
                collate_fn=collate_fn_default,
                sampler=valid_sampler
            )
            return train_loader, valid_loader, train_sampler, valid_sampler
        
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.config['dataset_params']['train_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['train_data_loader']['num_workers'],
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn_default
            )
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=self.config['dataset_params']['val_data_loader']['batch_size'],
                num_workers=self.config['dataset_params']['val_data_loader']['num_workers'],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn_default
            )
            return train_loader, valid_loader, None, None
    

    def _initOptimizer(self):
        if self.config['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params=self.model.parameters(), 
                lr=self.config['train_params']["learning_rate"],
                weight_decay=self.config['train_params']["weight_decay"]
                )
        elif self.config['train_params']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(), 
                lr=self.config['train_params']["learning_rate"],
                weight_decay=self.config['train_params']["weight_decay"],
                amsgrad=True)
        elif self.config['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.config['train_params']["learning_rate"],
                momentum=self.config['train_params']["momentum"],
                weight_decay=self.config['train_params']["weight_decay"],
                nesterov=self.config['train_params']["nesterov"]
                )
        else:
            raise NotImplementedError

        return optimizer

    
    def _initCriterion_ssc(self):
        criterion = {}
        criterion["CE_ssc_loss"] = CE_ssc_loss
        criterion["lovasz_loss"] = lovasz_softmax
        return criterion


    def _initCriterion_traj(self):
        criterion = {}
        criterion["trans_loss"] = TransLoss(loss_weight=0.01)
        criterion["rots_loss"] = RotationLoss(loss_weight=1.0)
        return criterion
        
    
    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def _computeClassifyLoss_ssc(self, pred, label):
        loss_ce = self.criterion_ssc['CE_ssc_loss'](
            pred, label, self.class_weights)
        loss_lovasz = self.criterion_ssc["lovasz_loss"](
            F.softmax(pred, dim=1), label.long(), ignore=255)
        
        if self.settings.n_gpus > 1 and not self.settings.distributed:
            loss_ce = loss_ce.mean()
            loss_lovasz = loss_lovasz.mean()

        return loss_ce + loss_lovasz

    def _computeTrajLoss(self, pred_delta_translations, targ_delta_translations, pred_relative_rotations, targ_relative_rotations):
        loss_trans = self.criterion_traj['trans_loss'](
            pred_delta_translations, targ_delta_translations)
        loss_rots = self.criterion_traj["rots_loss"](
            pred_relative_rotations, targ_relative_rotations)
        
        if self.settings.n_gpus > 1 and not self.settings.distributed:
            loss_trans = loss_trans.mean()
            loss_rots = loss_rots.mean()

        return loss_trans.sum(), loss_rots.sum()


    def train(self, epoch):
        if self.settings.distributed:
            torch.distributed.barrier()
            self.train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        loss_trans_meter = AverageMeter()
        loss_rots_meter = AverageMeter()

        t_start = time.time()
        self.model.train()

        for i, data_dict in enumerate(self.train_loader):
            t_process_start = time.time()

            ## cpu -> gpu
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()
            voxel_label = torch.cat((data_dict['his_occ'], data_dict['fut_occ']), dim=1)
            voxel_label = voxel_label.flatten(0, 1)

            prediction, pred_delta_translations, pred_relative_rotations = self.model(data_dict)
            prediction = prediction.flatten(0, 1)

            # calculate loss
            ssc_loss = self._computeClassifyLoss_ssc(
                prediction, voxel_label)
            trans_loss, rots_loss = self._computeTrajLoss(
                pred_delta_translations, 
                data_dict['fut_e2g_delta_translation'], 
                pred_relative_rotations, 
                data_dict['fut_e2g_relative_rotation'])
            
            total_loss = 0
            total_loss += ssc_loss
            total_loss += trans_loss
            total_loss += rots_loss

            if self.settings.n_gpus > 1:
                total_loss = total_loss.mean()

            self._backward(total_loss)
            self.scheduler.step()

            loss_meter.update(total_loss.item())
            loss_trans_meter.update(trans_loss.item())
            loss_rots_meter.update(rots_loss.item())
            loss = total_loss.mean()

            self.data_time.update(t_process_start - t_start)
            self.batch_time_train.update(time.time() - t_process_start)
            t_start = time.time()
            estimated_time = self.remain_time.calculate(epoch, i)
            
            if self.recorder is not None:
                for g in self.optimizer.param_groups:
                    lr = g['lr']
                    break
                log_str = ">>> Train [Epoch:{:03d}/{:03d}] [Iter:{:04d}/{:04d}] [DT:{data_time.val:.3f} ({data_time.avg:.3f})] [PT:{batch_time.val:.3f} ({batch_time.avg:.3f})] ".format(
                            epoch+1, self.config['train_params']['max_num_epochs'], i+1, len(self.train_loader), data_time=self.data_time, batch_time=self.batch_time_train)
                log_str += "[LR:{:0.5f}] [Loss:{:0.4f}] [Loss_trans:{:0.4f}] [Loss_rots:{:0.4f}] [RT:{}]".format(lr, loss.item(), trans_loss.item(), rots_loss.item(), estimated_time) 
                self.recorder.logger.info(log_str)
        
        if self.recorder is not None:
            # end of train
            log_str_end = ">>> Train End: [Loss:{:0.4f}] [Loss_trans:{:0.4f}] [Loss_rots:{:0.4f}]".format(
                loss_meter.avg, loss_trans_meter.avg, loss_rots_meter.avg)
            self.recorder.logger.info(log_str_end)

            self.recorder.tensorboard.add_scalar(
                tag="Train_Loss", scalar_value=loss_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="Train_Loss_translation", scalar_value=loss_trans_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="Train_Loss_rotation", scalar_value=loss_rots_meter.avg, global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="Train_lr", scalar_value=lr, global_step=epoch)

        result_metrics = {}
        return result_metrics


    def valid(self, epoch):
        if self.settings.distributed:
            torch.distributed.barrier()
        
        loss_meter = AverageMeter()
        loss_trans_meter = AverageMeter()
        loss_rots_meter = AverageMeter()

        self.model.eval()
        self.ssc_iou_meter_history.reset()
        self.sc_iou_meter_history.reset()
        self.ssc_iou_meter_future.reset()
        self.sc_iou_meter_future.reset()

        t_start = time.time()

        with torch.no_grad():
            for i, data_dict in enumerate(self.val_loader):

                ## cpu -> gpu
                for key in data_dict:
                    data_dict[key] = data_dict[key].cuda()
                start_frame, mid_frame, end_frame = \
                    0, data_dict['his_occ'].shape[1], data_dict['his_occ'].shape[1] + data_dict['fut_occ'].shape[1]
                voxel_label = torch.cat((data_dict['his_occ'], data_dict['fut_occ']), dim=1)
                voxel_label = voxel_label.flatten(0, 1)

                prediction, pred_delta_translations, pred_relative_rotations = self.model(data_dict)
                history_pred = prediction[:, start_frame:mid_frame].clone()
                future_pred = prediction[:, mid_frame:end_frame].clone()
                prediction = prediction.flatten(0, 1)

                # calculate loss
                ssc_loss = self._computeClassifyLoss_ssc(
                    prediction, voxel_label)
                trans_loss, rots_loss = self._computeTrajLoss(
                    pred_delta_translations, 
                    data_dict['fut_e2g_delta_translation'], 
                    pred_relative_rotations, 
                    data_dict['fut_e2g_relative_rotation'])
                
                total_loss = 0
                total_loss += ssc_loss
                total_loss += trans_loss
                total_loss += rots_loss

                if self.settings.n_gpus > 1:
                    total_loss = total_loss.mean()

                loss_meter.update(total_loss.item())
                loss_trans_meter.update(trans_loss.item())
                loss_rots_meter.update(rots_loss.item())
                loss = total_loss.mean()

                ## history occupancy
                history_label = data_dict['his_occ'].clone()
                ssc_label_eval = history_label.reshape(-1, 1)
                eval_mask = ssc_label_eval.lt(255).squeeze(1)
                ssc_label_eval = ssc_label_eval[eval_mask]

                ssc_logits_eval = history_pred.flatten(0, 1).argmax(dim=1).reshape(-1, 1)
                ssc_logits_eval = ssc_logits_eval[eval_mask]
                self.ssc_iou_meter_history.addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
                _, ssc_class_iou_h = self.ssc_iou_meter_history.getIoU()

                sc_logits_eval = ssc_logits_eval.lt(17).long()
                sc_label_eval = ssc_label_eval.lt(17).long()
                self.sc_iou_meter_history.addBatch(sc_logits_eval, sc_label_eval)
                _, sc_iou_h = self.sc_iou_meter_history.getIoU()

                ## future occupancy
                future_label = data_dict['fut_occ'].clone()
                ssc_label_eval = future_label.reshape(-1, 1)
                eval_mask = ssc_label_eval.lt(255).squeeze(1)
                ssc_label_eval = ssc_label_eval[eval_mask]

                ssc_logits_eval = future_pred.flatten(0, 1).argmax(dim=1).reshape(-1, 1)
                ssc_logits_eval = ssc_logits_eval[eval_mask]
                self.ssc_iou_meter_future.addBatch(ssc_logits_eval.long(), ssc_label_eval.long())
                _, ssc_class_iou_f = self.ssc_iou_meter_future.getIoU()

                sc_logits_eval = ssc_logits_eval.lt(17).long()
                sc_label_eval = ssc_label_eval.lt(17).long()
                self.sc_iou_meter_future.addBatch(sc_logits_eval, sc_label_eval)
                _, sc_iou_f = self.sc_iou_meter_future.getIoU()

                self.batch_time_valid.update(time.time() - t_start)
                t_start = time.time()

                if self.recorder is not None:
                    log_str = ">>> Valid [Iter:{:04d}/{:04d}] [PT:{batch_time.val:.3f} ({batch_time.avg:.3f})] ".format(
                                i+1, len(self.val_loader), batch_time=self.batch_time_valid)
                    log_str += "[Loss:{:0.4f}] [Loss_trans:{:0.4f}] [Loss_rots:{:0.4f}] [ssc_miou_h:{:0.4f}] [sc_iou_h:{:0.4f}] ".format(
                                loss.item(), trans_loss.item(), rots_loss.item(), ssc_class_iou_h[:17].mean().item(), sc_iou_h[1].item()) 
                    log_str += "[ssc_miou_f:{:0.4f}] [sc_iou_f:{:0.4f}]".format(
                                ssc_class_iou_f[:17].mean().item(), sc_iou_f[1].item()) 
                    self.recorder.logger.info(log_str)


            if self.recorder is not None:
                # end of valid 
                log_str_end = ">>> Valid End: [ssc_miou_h:{:0.4f}] [sc_iou_h:{:0.4f}] [ssc_miou_f:{:0.4f}] [sc_iou_f:{:0.4f}]".format(
                                ssc_class_iou_h[:17].mean().item(), sc_iou_h[1].item(), ssc_class_iou_f[:17].mean().item(), sc_iou_f[1].item()) 
                self.recorder.logger.info(log_str_end)

                for k, v in enumerate(self.cls_name):
                    valid_log = "class:{} [SSC_IoU_h:{:0.4f}] [SSC_IoU_f:{:0.4f}]".format(v, ssc_class_iou_h[k].item(), ssc_class_iou_f[k].item())
                    self.recorder.logger.info(valid_log) 
                    self.recorder.tensorboard.add_scalar(
                        tag="Valid_{:02d}_{}_SSC_IoU_History".format(k, v), 
                        scalar_value=ssc_class_iou_h[k].item(), global_step=epoch)
                    self.recorder.tensorboard.add_scalar(
                        tag="Valid_{:02d}_{}_SSC_IoU_Future".format(k, v), 
                        scalar_value=ssc_class_iou_f[k].item(), global_step=epoch)

                # tensorboard
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_Loss", scalar_value=loss_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_Loss_translation", scalar_value=loss_trans_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_Loss_rotation", scalar_value=loss_rots_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SC_IOU_History", scalar_value=sc_iou_h[1].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SSC_MIOU_History", scalar_value=ssc_class_iou_h[:17].mean().item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SC_IOU_Future", scalar_value=sc_iou_f[1].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="Valid_SSC_MIOU_Future", scalar_value=ssc_class_iou_f[:17].mean().item(), global_step=epoch)


            result_metrics = {}
            result_metrics['valid_sc_iou_history'] = sc_iou_h[1].item()
            result_metrics['valid_ssc_miou_history'] = ssc_class_iou_h[:17].mean().item()
            result_metrics['valid_sc_iou_future'] = sc_iou_f[1].item()
            result_metrics['valid_ssc_miou_future'] = ssc_class_iou_f[:17].mean().item()

        return result_metrics