# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from monai.metrics.utils import do_metric_reduction
from monai.utils.enums import MetricReduction
from tqdm import tqdm
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather, AverageMeter
import torch.utils.data.distributed
# from monai.transforms import SaveImage
import nibabel as nib
from monai.transforms import AsDiscrete
from monai.metrics.utils import get_surface_distance
from medpy.metric.binary import hd95, assd
from medpy.metric import binary


def train_epoch(model,
                loader,
                optimizer,
                epoch,
                args,
                loss_func,
                scheduler=None):  # Add scheduler argument here
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch in enumerate(loader):
        batch = {
            x: batch[x].to(torch.device('cuda', args.rank))
            for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
        }

        image = batch["image"]
        target = batch["label"]

        for param in model.parameters(): param.grad = None
        logits = model(image)

        # Check raw logits (before loss)
        with torch.no_grad():
            raw_preds = torch.sigmoid(logits)

        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()  # Update model parameters
        
        # Print only total grad norm of model
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5

        if scheduler is not None:
            scheduler.step()  # Update learning rate
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

class Trainer:
    def __init__(self, args,
                 train_loader,
                 loss_func,
                 validator=None,
                 ):
        pass
        self.args = args
        self.train_loader = train_loader
        self.validator = validator
        self.loss_func = loss_func

    def train(self, model,
              optimizer,
              scheduler=None,
              start_epoch=0,
              ):
        pass
        args = self.args
        train_loader = self.train_loader
        writer = None

        if args.logdir is not None and args.rank == 0:
            writer = SummaryWriter(log_dir=args.logdir)
            if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)

        val_acc_max_mean = 0.
        val_acc_max = 0.
        for epoch in range(start_epoch, args.max_epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            print(args.rank, time.ctime(), 'Epoch:', epoch)
            epoch_time = time.time()
            train_loss = train_epoch(model,
                                     train_loader,
                                     optimizer,
                                     epoch=epoch,
                                     args=args,
                                     loss_func=self.loss_func)
            if args.rank == 0:
                print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                      'time {:.2f}s'.format(time.time() - epoch_time))
            if args.rank==0 and writer is not None:
                writer.add_scalar('train_loss', train_loss, epoch)

            b_new_best = False
            if (epoch+1) % args.val_every == 0 and self.validator is not None:
                if args.distributed:
                    torch.distributed.barrier()
                epoch_time = time.time()

                _, val_avg_acc = self.validator.run()
                mean_dice = self.validator.metric_dice_avg(val_avg_acc)
                if args.rank == 0:
                    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                          'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time),
                          "mean_dice", mean_dice)
                    if writer is not None:
                        for name, value in val_avg_acc.items():
                            if "dice" in name.lower():
                                writer.add_scalar(name, value, epoch)
                        writer.add_scalar('mean_dice', mean_dice, epoch)

                    if mean_dice > val_acc_max_mean:
                        print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max_mean, mean_dice))
                        val_acc_max_mean = mean_dice
                        val_acc_max = val_avg_acc
                        b_new_best = True
                        if args.rank == 0 and args.logdir is not None:
                            save_checkpoint(model, epoch, args,
                                            best_acc=val_acc_max_mean,
                                            optimizer=optimizer,
                                            scheduler=scheduler)

                if args.rank == 0 and args.logdir is not None:
                    with open(os.path.join(args.logdir, "log.txt"), "a+") as f:
                        f.write(f"epoch:{epoch+1}, metric:{val_avg_acc}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, avg metric: {mean_dice}")
                        f.write("\n")
                        f.write(f"epoch:{epoch+1}, best metric:{val_acc_max}")
                        f.write("\n")
                        f.write(f"epoch: {epoch+1}, best avg metric: {val_acc_max_mean}")
                        f.write("\n")
                        f.write("*" * 20)
                        f.write("\n")

                    save_checkpoint(model,
                                    epoch,
                                    args,
                                    best_acc=val_acc_max,
                                    filename='model_final.pt')
                    if b_new_best:
                        print('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

            if scheduler is not None:
                scheduler.step()

        print('Training Finished !, Best Accuracy: ', val_acc_max)

        return val_acc_max

class Validator:
    def __init__(self,
                 args,
                 model,
                 val_loader,
                 class_list,
                 metric_functions,
                 sliding_window_infer=None,
                 post_label=None,
                 post_pred=None,

                 ) -> None:

        self.val_loader = val_loader
        self.sliding_window_infer = sliding_window_infer
        self.model = model
        self.args = args
        self.post_label = post_label
        self.post_pred = post_pred
        self.metric_functions = metric_functions
        self.class_list = class_list

    def metric_dice_avg(self, metric):
        metric_sum = 0.0
        c_nums = 0
        for m, v in metric.items():
            if "dice" in m.lower():
                metric_sum += v
                c_nums += 1

        return metric_sum / c_nums

    def is_best_metric(self, cur_metric, best_metric):

        best_metric_sum = self.metric_dice_avg(best_metric)
        metric_sum = self.metric_dice_avg(cur_metric)
        if best_metric_sum < metric_sum:
            return True

        return False

    def run(self):
        self.model.eval()
        args = self.args

        assert len(self.metric_functions[0]) == 2

        class_metric = []
        for m in self.metric_functions:
            for clas in self.class_list:
                class_metric.append(f"{clas}_{m[0]}")

        # Setup output path for predictions
        pred_dir = os.path.join(args.logdir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        all_case_metrics = []
        for idx, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
            batch = {
                x: batch[x].to(torch.device('cuda', args.rank))
                for x in batch if x not in ['fold', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'image_transforms', 'label_transforms']
            }

            label = batch["label"]

            with torch.no_grad():
                # ➕ Inference
                if self.sliding_window_infer is not None:
                    logits = self.sliding_window_infer(batch["image"], self.model)
                else:
                    logits = self.model(batch["image"])

                if self.post_label is not None:
                    label = self.post_label(label)

                probs = torch.sigmoid(logits)

                # Apply threshold using your post_pred_func (which includes sigmoid + thresholding)
                if self.post_pred is not None:
                    logits = self.post_pred(logits)

            # Save prediction using nibabel
            pred_np = logits.cpu().numpy()[0]  # shape: [C, H, W, D]
            if pred_np.shape[0] == 1:
                pred_np = pred_np[0]  # shape: [H, W, D]
            else:
                pred_np = np.transpose(pred_np, (1, 2, 3, 0))  # [H, W, D, C]

            # Default fallback values
            spacing = (1.0, 1.0, 1.0)
            affine = np.diag(spacing + (1.0,))

            meta = getattr(batch["image"], "meta", {})
            filename_or_obj = meta.get("filename_or_obj") or meta.get("image_path")
            if isinstance(filename_or_obj, list):
                filename_or_obj = filename_or_obj[0]

            if filename_or_obj:
                try:
                    nii = nib.load(filename_or_obj)
                    spacing = tuple(nii.header.get_zooms()[:3])
                except Exception:
                    pass

            if isinstance(meta, dict):
                aff = meta.get("affine", None)
                if isinstance(aff, torch.Tensor):
                    aff = aff.cpu().numpy()
                if isinstance(aff, np.ndarray):
                    if aff.ndim == 3:
                        aff = aff[0]
                    if aff.shape == (4, 4):
                        affine = aff

            filename = f"{idx:03d}.nii.gz"
            save_path = os.path.join(pred_dir, filename)
            nib.save(nib.Nifti1Image(pred_np.astype(np.uint8), affine), save_path)

            spacing = tuple(np.abs(np.diag(affine))[:3])

            pred_bin = logits.cpu().numpy().astype(bool)
            label_bin = label.cpu().numpy().astype(bool)

            tp = np.logical_and(pred_bin, label_bin).sum()
            tn = np.logical_and(~pred_bin, ~label_bin).sum()
            fp = np.logical_and(pred_bin, ~label_bin).sum()
            fn = np.logical_and(~pred_bin, label_bin).sum()

            iou = tp / (tp + fp + fn + 1e-8)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

            pred_bin_metric = np.squeeze(pred_bin).astype(np.uint8)
            label_bin_metric = np.squeeze(label_bin).astype(np.uint8)

            try:
                hd = binary.hd95(pred_bin_metric, label_bin_metric, voxelspacing=spacing)
                as_sd = binary.assd(pred_bin_metric, label_bin_metric, voxelspacing=spacing)
            except Exception as e:
                print(f"MedPy metric error (case {idx}): {e}")
                hd = -1.0
                as_sd = -1.0

            all_case_metrics.append({
                "ASSD": as_sd,
                "Dice": dice,
                "FN": fn,
                "FP": fp,
                "HD95": hd,
                "IoU": iou,
                "TN": tn,
                "TP": tp,
                "n_pred": pred_bin.sum(),
                "n_ref": label_bin.sum()
            })

        avg_metrics = {}
        for key in all_case_metrics[0].keys():
            avg_metrics[key] = float(np.mean([case[key] for case in all_case_metrics]))

        return all_case_metrics, avg_metrics