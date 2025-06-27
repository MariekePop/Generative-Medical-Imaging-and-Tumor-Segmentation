import argparse
from utils.data_utils import get_loader
from medical.trainer import Trainer, Validator
from monai.inferers import SlidingWindowInferer
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import numpy as np
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from medical.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss
from medical.model.nested_former import NestedFormer
import os
import json

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')
parser.add_argument('--model_name', default="swinunetr", help='the model will be trained')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--fold', default=0, type=int, help='data fold')
parser.add_argument('--pretrain_model_path', default='./model.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--data_dir', default='/mnt/datasets/brats2020/MICCAI_BraTS2020_TrainingData', type=str, help='dataset directory')
parser.add_argument('--json_list', default='./brats2020_datajson.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=4, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization name')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=256, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=256, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=64, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.25, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--test_only', action='store_true', help='run inference on the validation or test set')


def post_pred_func(pred):
    return (torch.sigmoid(pred) > 0.5).float()


def main():
    args = parser.parse_args()
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    train_loader, val_loader = get_loader(args)

    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    model = NestedFormer(model_num=args.in_channels,
                         out_channels=args.out_channels,
                         image_size=inf_size,
                         window_size=(4, 4, 4),
                         )

    window_infer = SlidingWindowInferer(roi_size=inf_size,
                                        sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        model.load_state_dict(new_state_dict, strict=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True
                                                          )
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    dice_metric = DiceMetric(include_background=True,
                             reduction=MetricReduction.MEAN_BATCH,
                             get_not_nans=True)

    validator = Validator(args,
                          model,
                          val_loader,
                          # class_list=("TC", "WT", "ET"),
                          class_list=("tumor",),
                          metric_functions=[["dice", dice_metric]],
                          sliding_window_infer=window_infer,
                          post_label=None,
                          post_pred=post_pred_func)
    
    if args.test_only:
        print("Running validation/testing...")
        # results = validator.run()
        results, _ = validator.run()  # Keep your existing nnUNet-style block untouched
        if args.rank == 0:
            print("Validation results:")
            for i, case_metrics in enumerate(results):
                print(f"Case {i}:")
                for metric, value in case_metrics.items():
                    print(f"  {metric}: {value}")

            # Build nnUNet-style results format
            metric_per_case = []
            for idx, metrics in enumerate(results):
                entry = {
                    "metrics": {
                        "1": {k: float(v) for k, v in metrics.items()}
                    },
                    "prediction_file": os.path.join(args.logdir, "predictions", f"{idx:03d}.nii.gz"),
                    "reference_file": os.path.join("labelsTs", f"{idx:03d}.nii.gz")  # adjust if needed
                }
                metric_per_case.append(entry)

            # Compute foreground_mean
            metric_keys = results[0].keys()
            means = {k: np.mean([float(case[k]) for case in results]) for k in metric_keys}
            foreground_mean = {k: float(v) for k, v in means.items()}

            formatted_results = {
                "foreground_mean": foreground_mean,
                "mean": {
                    "1": foreground_mean
                },
                "metric_per_case": metric_per_case
            }

            # Save final JSON
            os.makedirs(args.logdir, exist_ok=True)
            with open(os.path.join(args.logdir, "results.json"), "w") as f:
                json.dump(formatted_results, f, indent=4)

        return


    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    trainer = Trainer(args,
                      train_loader,
                      validator=validator,
                      loss_func=dice_loss,
                      )

    best_acc = trainer.train(model,
                  optimizer=optimizer,
                  scheduler=scheduler)

    return best_acc

if __name__ == '__main__':
    main()
