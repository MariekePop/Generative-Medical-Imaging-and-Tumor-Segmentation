import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import random
import numpy as np
import torch
import torchvision
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from torch._dynamo import OptimizedModule

from nnseq2seq.networks.ema import EMAModel
from nnseq2seq.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnseq2seq.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnseq2seq.inference.predict_from_raw_data import nnSeq2SeqPredictor
from nnseq2seq.inference.sliding_window_prediction import compute_gaussian
from nnseq2seq.paths import nnSeq2Seq_preprocessed, nnSeq2Seq_results
from nnseq2seq.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnseq2seq.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnseq2seq.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnseq2seq.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnseq2seq.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnseq2seq.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnseq2seq.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform, CopyDataTransform
from nnseq2seq.training.dataloading.data_loader_2d import nnSeq2SeqDataLoader2D
from nnseq2seq.training.dataloading.data_loader_3d import nnSeq2SeqDataLoader3D
from nnseq2seq.training.dataloading.nnseq2seq_dataset import nnSeq2SeqDataset
from nnseq2seq.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnseq2seq.training.logging.nnseq2seq_logger import nnSeq2SeqLogger
from nnseq2seq.training.loss.compound_losses import L1_SSIM_and_Perceptual_loss, DC_and_CE_loss
from nnseq2seq.training.loss.deep_supervision import DeepSupervisionWrapper
from nnseq2seq.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnseq2seq.training.loss.metrics import torch_PSNR
from nnseq2seq.training.loss.adversarial_loss import GANLoss
from nnseq2seq.training.lr_scheduler.polylr import PolyLRScheduler, WarmupCosineLRScheduler
from nnseq2seq.utilities.collate_outputs import collate_outputs
from nnseq2seq.utilities.crossval_split import generate_crossval_split
from nnseq2seq.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnseq2seq.utilities.file_path_utilities import check_workers_alive_and_busy
from nnseq2seq.utilities.get_network_from_plans import get_network_from_plans
from nnseq2seq.utilities.helpers import empty_cache, dummy_context
from nnseq2seq.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnseq2seq.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch.nn.functional as F
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class nnSeq2SeqTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)

        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # print what device we are using
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]

        ###  Saving all the init args into class variables for later access
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset

        ### Setting all the folder names. We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = join(nnSeq2Seq_preprocessed, self.plans_manager.dataset_name) \
            if nnSeq2Seq_preprocessed is not None else None
        self.output_folder_base = join(nnSeq2Seq_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnSeq2Seq_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')

        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        # unlike the previous nnseq2seq folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnSeq2Seq_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None

        ### Some hyperparameters for you to fiddle with
        self.train_segmentation_only = False
        self.train_translation_only = False
        assert (self.train_segmentation_only and self.train_translation_only)==False, 'Cannot set both train_segmentation_only and train_translation_only as True at the same time.'

        self.initial_lr = 2e-4
        self.weight_decay = 0.05 #3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        if self.train_segmentation_only:
            self.num_epochs = 1000
            self.num_epochs_for_pretrain = 1000
        else:
            self.num_epochs = 2000
            self.num_epochs_for_pretrain = 1000
        self.current_epoch = 0
        self.enable_deep_supervision = True

        ### Dealing with labels/regions
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions

        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.network_ema = None
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.optimizer_d = self.lr_scheduler = None
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.loss = None  # -> self.initialize
        self.seg_loss = None
        self.gan = GANLoss('hinge').to(device=self.device)
        self.img_dis = None
        self.tgt_seq_pool = None

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnSeq2SeqLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        self._set_batch_size_and_oversample()

        self.was_initialized = False

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnSeq2Seq:\n"
                               "[1] Han L, Tan T, Zhang T, et al. "
                               "Synthesis-based imaging-differentiation representation learning for multi-sequence 3D/4D MRI[J]. "
                               "Medical Image Analysis, 2024, 92: 103044.\n"
                               "[2] Han L, Zhang T, Huang Y, et al. "
                               "An Explainable Deep Framework: Towards Task-Specific Fusion for Multi-to-One MRI Synthesis[C]. "
                               "International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023: 45-55.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.network_ema = EMAModel(self.network.parameters())
            self.network_ema.to(device=self.device)
            self.optimizer, self.optimizer_d, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss, self.seg_loss = self._build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _do_i_compile(self):
        return ('nnSeq2Seq_compile' in os.environ.keys()) and (os.environ['nnSeq2Seq_compile'].lower() in ('true', '1', 't'))

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss', ]:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network', ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ['dataloader_train', 'dataloader_val']:
                    if hasattr(getattr(self, k), 'generator'):
                        dct[k + '.generator'] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), 'num_processes'):
                        dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), 'transform'):
                        dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnSeq2Seq default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnSeq2SeqTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            #deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            #    self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
            deep_supervision_scales = [1/2**i for i, _ in enumerate(
                self.configuration_manager.network_arch_init_kwargs['image_decoder']['conv_kernel'])]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            self.batch_size = self.configuration_manager.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                    'GPUs... Duh.'

            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size

            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

            # This is how oversampling is determined in DataLoader
            # round(self.batch_size * (1 - self.oversample_foreground_percent))
            # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
            # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
            # compute oversample
            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]

            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self):
        # if self.label_manager.has_regions:
        #     loss = DC_and_BCE_loss({},
        #                            {'batch_dice': self.configuration_manager.batch_dice,
        #                             'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
        #                            use_ignore_label=self.label_manager.ignore_label is not None,
        #                            dice_class=MemoryEfficientSoftDiceLoss)
        # else:
        #     loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
        #                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
        #                           ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        spatial_dims = 2 if self.configuration_name=='2d' else 3
        if spatial_dims==2:
            weight_perceptual = 0.01
            scale_factor = 1
        else:
            weight_perceptual = 0.01
            scale_factor = 1
        loss = L1_SSIM_and_Perceptual_loss(self.device, weight_l1=10, weight_ssim=1, weight_perceptual=weight_perceptual, spatial_dims=spatial_dims)
        loss_seg = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                            'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'reduction': 'none'}, weight_ce=1, weight_dice=1,
                                            ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([(scale_factor ** i) for i in range(len(deep_supervision_scales))])
            #if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
            #    weights[-1] = 1e-6
            #else:
            #    weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
            loss_seg = DeepSupervisionWrapper(loss_seg, weights)
        return loss, loss_seg

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnseq2seq for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                            momentum=0.99, nesterov=True)
        optimizer = torch.optim.AdamW(
            list(self.network.image_encoder.parameters()) + \
            list(self.network.hyper_decoder.parameters()) + \
            list(self.network.segmentor.parameters()), lr=self.initial_lr,
            betas=(0.9, 0.95), weight_decay=self.weight_decay)
        optimizer_d = torch.optim.AdamW(
            self.network.discriminator.parameters(), lr=self.initial_lr,
            betas=(0.9, 0.95), weight_decay=self.weight_decay)
        #lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = WarmupCosineLRScheduler(optimizer, self.initial_lr, self.num_epochs,
            min_lr=1e-7, T_multi=1, N_cycle=2, warmup_epochs=10, warmup_initial_lr=1e-7)
        return optimizer, optimizer_d, lr_scheduler

    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnSeq2Seq_compile is enabled!")
            return

        if self.local_rank == 0:
            try:
                # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
                # pip install git+https://github.com/saugatkandel/hiddenlayer.git

                # from torchviz import make_dot
                # # not viable.
                # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
                #                                         *self.configuration_manager.patch_size),
                #                                        device=self.device)))).render(
                #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
                # self.optimizer.zero_grad()

                # broken.

                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)

                # self.print_to_log_file("\nprinting the network instead:\n")
                # self.print_to_log_file(self.network)
                # self.print_to_log_file("\n")
            finally:
                empty_cache(self.device)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnSeq2Seq will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnSeq2Seq is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnSeq2Seq will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnSeq2SeqDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.keys())))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnSeq2SeqDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnSeq2SeqDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnSeq2SeqDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnSeq2SeqDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnSeq2SeqDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnSeq2SeqDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="nearest", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())
        
        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))
        
        tr_transforms.append(CopyDataTransform(input_key='data', output_key='ori_data'))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['ori_data', 'data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnSeq2Seq. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.hyper_decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=7)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def random_select_seq_code(self, properties):
        output_code = []
        for bid in range(len(properties)):
            seq_id = random.choice([i for i in range(self.num_input_channels) if i not in properties[bid]['input_domain']])
            output_code.append(seq_id)
        output_code = torch.from_numpy(np.array(output_code)).to(dtype=torch.int64)
        return output_code

    def random_select_available_seq(self, data, properties):
        output_data = []
        output_code = []
        other_code = []
        output_mask = []
        output_atlas = []
        flag_atlas = []
        for bid in range(len(properties)):
            if self.tgt_seq_pool is None:
                self.tgt_seq_pool = [0 for i in range(self.num_input_channels)]
            inner_list = [v if i in properties[bid]['available_channel'] and i not in properties[bid]['input_domain'] else 1e+9 for i, v in enumerate(self.tgt_seq_pool)]
            
            if random.random()>0.5:
                seq_id = random.choice(properties[bid]['available_channel'])
            else:
                seq_id = np.int64(inner_list.index(min(inner_list)))
            output_data.append(data[bid:bid+1, seq_id:seq_id+1])
            output_code.append(seq_id)
            self.tgt_seq_pool[seq_id] += 1
            other_code.append(random.choice([i for i in range(self.num_input_channels) if i!=seq_id]))
            
            # calculate overlap mask, but it's hard because some image do not have zero background
            fg = torch.ones_like(data[0:1, 0:1])
            bg = torch.ones_like(data[0:1, 0:1])
            for seq_id in properties[bid]['available_channel']:
                fg *= (data[bid:bid+1, seq_id:seq_id+1]>0)
                bg *= (data[bid:bid+1, seq_id:seq_id+1]<=0)
            output_mask.append(fg+bg)
            atlas_id = 0 if len(properties[bid]['atlas_domain'])==0 else random.choice(properties[bid]['atlas_domain'])
            output_atlas.append(data[bid:bid+1, atlas_id:atlas_id+1])
            flag_atlas.append(1 if atlas_id in properties[bid]['available_channel'] else 0)

        output_data = torch.cat(output_data, dim=0)
        output_code = torch.from_numpy(np.array(output_code))
        other_code = torch.from_numpy(np.array(other_code)).to(dtype=output_code.dtype)
        output_mask = torch.cat(output_mask, dim=0)
        output_atlas = torch.cat(output_atlas, dim=0)
        flag_atlas = torch.from_numpy(np.array(flag_atlas))
        return output_data, output_code, other_code, output_mask, output_atlas, flag_atlas
    
    def random_select_available_multiseqs(self, properties, tgt_id):
        output_code = []
        output_code_all = []
        for bid in range(len(properties)):
            available_input = [i for i in properties[bid]['available_channel'] if i not in properties[bid]['output_domain']]
            seqs_id = np.sort(random.sample(available_input, random.choice(
                [i for i in range(1, max(2, len(available_input)))])))
            output_code.append([1 if i in seqs_id and (i!=tgt_id[bid] or len(seqs_id)==1) else 0 for i in range(self.num_input_channels)])
            output_code_all.append([1 if i in available_input else 0 for i in range(self.num_input_channels)])
            
        output_code = torch.from_numpy(np.array(output_code))
        output_code_all = torch.from_numpy(np.array(output_code_all))
        return output_code_all, output_code
    
    def generate_seg_weights(self, properties):
        output_weight = []
        for bid in range(len(properties)):
            if 'foreground' in self.dataset_json['labels'].keys() and len(properties[bid]['class_locations'][self.dataset_json['labels']['foreground']])>0:
                output_weight.append(0)
            else:
                output_weight.append(1)
        output_weight = torch.from_numpy(np.array(output_weight))
        return output_weight

    def train_step(self, batch_id: int, batch: dict) -> dict:
        ori_data = torch.clamp(batch['ori_data'], min=0)
        aug_data = batch['data']
        target = batch['target']
        properties = batch['properties']
        seg_weights = self.generate_seg_weights(properties)

        tgt_data, tgt_code_int64, _, tgt_mask, atlas_data, flag_atlas = self.random_select_available_seq(ori_data, properties)
        src_all_code, src_code = self.random_select_available_multiseqs(properties, tgt_code_int64)
        rand_code_int64 = self.random_select_seq_code(properties)

        src_data = aug_data.to(self.device, non_blocking=True)
        tgt_data = tgt_data.to(self.device, non_blocking=True)
        tgt_mask = tgt_mask.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        src_all_code = src_all_code.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        src_code = src_code.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        tgt_code = F.one_hot(tgt_code_int64, num_classes=self.num_input_channels).to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        rand_code = F.one_hot(rand_code_int64, num_classes=self.num_input_channels).to(self.device, dtype=src_data.dtype, non_blocking=True)

        atlas_data = atlas_data.to(self.device, non_blocking=True)
        flag_atlas = flag_atlas.to(self.device, dtype=atlas_data.dtype, non_blocking=True).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        seg_weights = seg_weights.to(self.device, dtype=src_data.dtype, non_blocking=True)

        if self.img_dis is None:
            self.img_dis = torch.zeros([self.num_input_channels]+list(tgt_data.shape[1:])).to(self.device, non_blocking=True)
        img_rand_dis = []
        for rand_id in range(rand_code_int64.shape[0]):
            img_rand_dis.append(self.img_dis[rand_code_int64[rand_id]])
        img_rand_dis = torch.stack(img_rand_dis, dim=0).detach()

        # dis
        if self.current_epoch>self.num_epochs_for_pretrain:
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                output_src2rand_all, output_src2rand_sub, _, _ = self.network(src_data, src_all_code, src_code, rand_code, with_latent=False)
            pred_real = self.network.discriminator(img_rand_dis, rand_code)
            if not self.enable_deep_supervision:
                output_src2rand_all = [output_src2rand_all]
                output_src2rand_sub = [output_src2rand_sub]
            pred_fake1 = self.network.discriminator(torch.clamp(output_src2rand_all[0].detach(), min=0), rand_code)
            pred_fake2 = self.network.discriminator(torch.clamp(output_src2rand_sub[0].detach(), min=0), rand_code)
            ld = self.gan(pred_real, True) + 0.5*self.gan(pred_fake1, False) + 0.5*self.gan(pred_fake2, False)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(self.network.discriminator.parameters(), 12)
            self.optimizer_d.step()

        # adv
        self.optimizer.zero_grad(set_to_none=True)
        
        #with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        output_src2tgt_all, output_src2tgt_sub, latent_src_all, latent_src_sub, latent_src_all_seg, latent_src_sub_seg, vq_loss_src, fusion_all, fusion_subgroup = self.network(src_data, src_all_code, src_code, tgt_code, with_latent=True)
        mask_src_all = self.network.segmentor(latent_src_all_seg)
        mask_src_sub = self.network.segmentor(latent_src_sub_seg)

        # del data
        #int_datas = [F.interpolate(int_data, scale_factor=1/2**i, mode='nearest') for i in range(len(target))]
        if self.enable_deep_supervision:
            tgt_datas = [F.interpolate(tgt_data, scale_factor=1/2**i, mode='bilinear' if len(tgt_data.shape)==4 else 'trilinear') for i in range(len(target))]
            tgt_masks = [F.interpolate(tgt_mask, scale_factor=1/2**i, mode='nearest') for i in range(len(target))]
            if self.train_segmentation_only:
                l = self.seg_loss(mask_src_all, target, [seg_weights for _ in target]) + \
                    self.seg_loss(mask_src_sub, target, [seg_weights for _ in target])
            elif self.train_translation_only:
                l = self.loss(output_src2tgt_all, tgt_datas, tgt_masks) + \
                    self.loss(output_src2tgt_sub, tgt_datas, tgt_masks)
            else:
                l = self.loss(output_src2tgt_all, tgt_datas, tgt_masks) + \
                    self.loss(output_src2tgt_sub, tgt_datas, tgt_masks) + \
                    self.seg_loss(mask_src_all, target, [seg_weights for _ in target]) + \
                    self.seg_loss(mask_src_sub, target, [seg_weights for _ in target])
        else:
            if self.train_segmentation_only:
                l = self.seg_loss(0, mask_src_all, target, seg_weights) + \
                    self.seg_loss(0, mask_src_sub, target, seg_weights)
            elif self.train_translation_only:
                l = self.loss(0, output_src2tgt_all, tgt_data, tgt_mask) + \
                    self.loss(0, output_src2tgt_sub, tgt_data, tgt_mask)
            else:
                l = self.loss(0, output_src2tgt_all, tgt_data, tgt_mask) + \
                    self.loss(0, output_src2tgt_sub, tgt_data, tgt_mask) + \
                    self.seg_loss(0, mask_src_all, target, seg_weights) + \
                    self.seg_loss(0, mask_src_sub, target, seg_weights)

        l = l + vq_loss_src   
        if not self.train_segmentation_only:
            l = l + \
                nn.MSELoss()(fusion_all[1], latent_src_all[-1].detach()) + \
                nn.MSELoss()(fusion_subgroup[1], latent_src_sub[-1].detach()) + \
                nn.MSELoss()(fusion_all[0]*flag_atlas*tgt_mask, atlas_data*flag_atlas*tgt_mask) + \
                nn.MSELoss()(fusion_subgroup[0]*flag_atlas*tgt_mask, atlas_data*flag_atlas*tgt_mask)

            
        if self.current_epoch>self.num_epochs_for_pretrain:
            output_src2rand_all, _ = self.network.hyper_decoder(latent_src_all, rand_code)
            output_src2rand_sub, _ = self.network.hyper_decoder(latent_src_sub, rand_code)
            if not self.enable_deep_supervision:
                output_src2rand_all = [output_src2rand_all]
                output_src2rand_sub = [output_src2rand_sub]
            pred_fake1 = self.network.discriminator(torch.clamp(output_src2rand_all[0], min=0), rand_code)
            pred_fake2 = self.network.discriminator(torch.clamp(output_src2rand_sub[0], min=0), rand_code)
            l += (self.gan(pred_fake1, True) + self.gan(pred_fake2, True) + \
                nn.MSELoss()(output_src2rand_sub[0], torch.clamp(output_src2rand_all[0].detach(), min=0)) + \
                nn.MSELoss()(output_src2rand_all[0], torch.clamp(output_src2rand_sub[0].detach(), min=0)))
        else:
            if not self.enable_deep_supervision:
                output_src2rand_all = [output_src2tgt_all]
                output_src2rand_sub = [output_src2tgt_sub]
            else:
                output_src2rand_all = output_src2tgt_all
                output_src2rand_sub = output_src2tgt_sub
            
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.image_encoder.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network.hyper_decoder.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network.segmentor.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.image_encoder.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network.hyper_decoder.parameters(), 12)
            torch.nn.utils.clip_grad_norm_(self.network.segmentor.parameters(), 12)
            self.optimizer.step()
        
        self.network_ema.step(self.network.parameters())
        
        for tgt_id in range(tgt_code_int64.shape[0]):
            if torch.std(tgt_data[tgt_id])>0:
                self.img_dis[tgt_code_int64[tgt_id]] = tgt_data[tgt_id]
            

        vis_path = os.path.join(self.output_folder, 'visualization')
        os.makedirs(vis_path, exist_ok=True)
        vis_save_path = os.path.join(vis_path, 'epoch_{}.jpg'.format(self.current_epoch))
        if batch_id==0:
            if not self.enable_deep_supervision:
                output_src2tgt_all = [output_src2tgt_all]
                output_src2tgt_sub = [output_src2tgt_sub]
                mask_src_all = [mask_src_all]
                mask_src_sub = [mask_src_sub]
                target = [target]

            with torch.no_grad():
                max_label = mask_src_all[0].shape[1] - 1
                mask_src_all = [torch.argmax(m, dim=1, keepdim=True)/max_label for m in mask_src_all]
                mask_src_sub = [torch.argmax(m, dim=1, keepdim=True)/max_label for m in mask_src_sub]
                if len(src_data.shape)==5:
                    sd = src_data.shape[2]//2
                    vimage = torch.stack([
                        tgt_data[:,:,sd], output_src2tgt_all[0][:,:,sd], output_src2tgt_sub[0][:,:,sd], torch.abs(output_src2tgt_all[0]-output_src2tgt_sub[0])[:,:,sd],
                        img_rand_dis[:,:,sd], output_src2rand_all[0][:,:,sd], output_src2rand_sub[0][:,:,sd], torch.abs(output_src2rand_all[0]-output_src2rand_sub[0])[:,:,sd],
                        target[0][:,:,sd]/max_label, mask_src_all[0][:,:,sd], mask_src_sub[0][:,:,sd], tgt_mask[:,:,sd],
                        tgt_mask[:,:,sd], fusion_all[0][:,0:1,sd], fusion_all[0][:,1:2,sd], fusion_all[0][:,2:3,sd],
                    ], dim=1).reshape(-1,1,*(src_data.shape[3:]))
                    vimage = torch.clamp(vimage, min=0, max=1)
                    torchvision.utils.save_image(vimage, vis_save_path)
                    torchvision.utils.save_image(self.img_dis[:,:,sd], os.path.join(vis_path, 'discriminator.jpg'))
                    for vi, (lat_all, lat_sub) in enumerate(zip(latent_src_all, latent_src_sub)):
                        sd = lat_all.shape[2]//2
                        vimage = torch.stack([
                            lat_all[:,:,sd], lat_sub[:,:,sd]
                        ], dim=1).reshape(-1,lat_all.shape[1],*(lat_all.shape[3:]))
                        torchvision.utils.save_image(vimage, os.path.join(vis_path, 'latent_space_{}.jpg'.format(vi)))
                    for vi, (out_all, out_sub, out_tgt, m_all, m_sub, m_tgt) in enumerate(zip(output_src2tgt_all, output_src2tgt_sub, tgt_datas, mask_src_all, mask_src_sub, target)):
                        sd = out_all.shape[2]//2
                        vimage = torch.stack([
                            out_all[:,:,sd], out_tgt[:,:,sd], out_sub[:,:,sd], out_tgt[:,:,sd], m_all[:,:,sd], m_tgt[:,:,sd]/max_label, m_sub[:,:,sd], m_tgt[:,:,sd]/max_label
                        ], dim=1).reshape(-1,1,*(out_all.shape[3:]))
                        vimage = torch.clamp(vimage, min=0, max=1)
                        torchvision.utils.save_image(vimage, os.path.join(vis_path, 'deep_{}.jpg'.format(vi)))
                else:
                    vimage = torch.stack([
                        tgt_data, output_src2tgt_all[0], output_src2tgt_sub[0], torch.abs(output_src2tgt_all[0]-output_src2tgt_sub[0]),
                        img_rand_dis, output_src2rand_all[0], output_src2rand_sub[0], torch.abs(output_src2rand_all[0]-output_src2rand_sub[0]),
                        target[0]/max_label, mask_src_all[0], mask_src_sub[0], tgt_mask,
                        tgt_mask, fusion_all[0][:, 0:1], fusion_all[0][:, 1:2], fusion_all[0][:, 2:3],
                    ], dim=1).reshape(-1,1,*(src_data.shape[2:]))
                    vimage = torch.clamp(vimage, min=0, max=1)
                    torchvision.utils.save_image(vimage, vis_save_path)

                    torchvision.utils.save_image(self.img_dis, os.path.join(vis_path, 'discriminator.jpg'))

                    vimage = torch.stack([
                            fusion_all[0], fusion_all[1], fusion_subgroup[0], fusion_subgroup[1], 
                        ], dim=1).reshape(-1,*(fusion_all[0].shape[1:]))
                    vimage = torch.clamp(vimage, min=0, max=1)
                    torchvision.utils.save_image(vimage, os.path.join(vis_path, 'fusion.jpg'))

                    for vi, (lat_all, lat_sub) in enumerate(zip(latent_src_all, latent_src_sub)):
                        vimage = torch.stack([
                            lat_all, lat_sub
                        ], dim=1).reshape(-1,*(lat_all.shape[1:]))
                        torchvision.utils.save_image(vimage, os.path.join(vis_path, 'latent_space_{}.jpg'.format(vi)))
                    
                    if self.enable_deep_supervision:
                        for vi, (out_all, out_sub, out_tgt, m_all, m_sub, m_tgt) in enumerate(zip(output_src2tgt_all, output_src2tgt_sub, tgt_datas, mask_src_all, mask_src_sub, target)):
                            vimage = torch.stack([
                                out_all, out_tgt, out_sub, out_tgt, m_all, m_tgt/max_label, m_sub, m_tgt/max_label
                            ], dim=1).reshape(-1,1,*(out_all.shape[2:]))
                            vimage = torch.clamp(vimage, min=0, max=1)
                            torchvision.utils.save_image(vimage, os.path.join(vis_path, 'deep_{}.jpg'.format(vi)))
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        properties = batch['properties']
        seg_weights = self.generate_seg_weights(properties)

        tgt_data, tgt_code_int64, _, tgt_mask, _, _ = self.random_select_available_seq(data, properties)
        src_all_code, src_code = self.random_select_available_multiseqs(properties, tgt_code_int64)
        
        src_data = data.to(self.device, non_blocking=True)
        tgt_data = tgt_data.to(self.device, non_blocking=True)
        tgt_mask = tgt_mask.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        src_all_code = src_all_code.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        src_code = src_code.to(self.device, dtype=tgt_data.dtype, non_blocking=True)
        tgt_code = F.one_hot(tgt_code_int64, num_classes=self.num_input_channels).to(self.device, dtype=tgt_data.dtype, non_blocking=True)

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        seg_weights = seg_weights.to(self.device, dtype=src_data.dtype, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            _, output, _, _, latent_src_all_seg, _, _, _, _ = self.network(src_data, src_all_code, src_code, tgt_code,
                                                                           with_latent=True,
                                                                           latent_focal='dispersion' if self.network.hyper_decoder.focal_mode=='focal_mix' else None)
            mask_src_all = self.network.segmentor(latent_src_all_seg)
            del data
            if self.train_segmentation_only:
                l = self.seg_loss(mask_src_all, target, [seg_weights for _ in target])
            else:
                l = self.loss(
                    output,
                    [F.interpolate(tgt_data, scale_factor=1/2**i, mode='bilinear' if len(tgt_data.shape)==4 else 'trilinear') for i in range(len(target))],
                    [F.interpolate(tgt_mask, scale_factor=1/2**i, mode='bilinear' if len(tgt_data.shape)==4 else 'trilinear') for i in range(len(target))])

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = torch.clamp(output[0], 0, 1)
            tgt_data = torch.clamp(tgt_data, 0, 1)
            #target = target[0]
        
        if self.train_segmentation_only:
            psnr = -l
        else:
            psnr = torch_PSNR(tgt_data*tgt_mask, output*tgt_mask, data_range=1)

        # # the following is needed for online evaluation. Fake dice (green line)
        # axes = [0] + list(range(2, output.ndim))

        # if self.label_manager.has_regions:
        #     predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        # else:
        #     # no need for softmax
        #     output_seg = output.argmax(1)[:, None]
        #     predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        #     predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        #     del output_seg

        # if self.label_manager.has_ignore_label:
        #     if not self.label_manager.has_regions:
        #         mask = (target != self.label_manager.ignore_label).float()
        #         # CAREFUL that you don't rely on target after this line!
        #         target[target == self.label_manager.ignore_label] = 0
        #     else:
        #         mask = 1 - target[:, -1:]
        #         # CAREFUL that you don't rely on target after this line!
        #         target = target[:, :-1]
        # else:
        #     mask = None

        # tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        # tp_hard = tp.detach().cpu().numpy()
        # fp_hard = fp.detach().cpu().numpy()
        # fn_hard = fn.detach().cpu().numpy()
        # if not self.label_manager.has_regions:
        #     # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        #     # (softmax training) there needs tobe one output for the background. We are not interested in the
        #     # background Dice
        #     # [1:] in order to remove background
        #     tp_hard = tp_hard[1:]
        #     fp_hard = fp_hard[1:]
        #     fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'psnr': psnr.item()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        #tp = np.sum(outputs_collated['tp_hard'], 0)
        #fp = np.sum(outputs_collated['fp_hard'], 0)
        #fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            #tps = [None for _ in range(world_size)]
            #dist.all_gather_object(tps, tp)
            #tp = np.vstack([i[None] for i in tps]).sum(0)

            #fps = [None for _ in range(world_size)]
            #dist.all_gather_object(fps, fp)
            #fp = np.vstack([i[None] for i in fps]).sum(0)

            #fns = [None for _ in range(world_size)]
            #dist.all_gather_object(fns, fn)
            #fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            psnr_val = [None for _ in range(world_size)]
            dist.all_gather_object(psnr_val, outputs_collated['psnr'])
            psnr_here = np.vstack([i[None] for i in psnr_val]).sum(0)
        else:
            loss_here = np.mean(outputs_collated['loss'])
            psnr_here = np.mean(outputs_collated['psnr'])

        #global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        #mean_fg_dice = np.nanmean(global_dc_per_class)
        #self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        #self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('PSNR', psnr_here, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('PSNR', np.round(self.logger.my_fantastic_logging['PSNR'][-1], decimals=4))
        # self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
        #                                        self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_PSNR'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_PSNR'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo PSNR: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                self.network_ema.store(self.network.parameters())
                self.network_ema.copy_to(self.network.parameters())
                
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)

                self.network_ema.restore(self.network.parameters())
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            # checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)

        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key in self.network.state_dict().keys():
                new_state_dict[key] = value
            elif key.startswith('module.') and key[7:] in self.network.state_dict().keys():
                key = key[7:]
                new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnSeq2Seq to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        predictor = nnSeq2SeqPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(8) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = nnSeq2SeqDataset(self.preprocessed_dataset_folder, val_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                        num_images_properties_loading_threshold=0)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded:
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                for src_seq in properties['available_channel']:
                    for tgt_seq in properties['available_channel']:
                        tgt_code = F.one_hot(torch.from_numpy(np.array([tgt_seq])),
                                             num_classes=self.num_input_channels).to(self.device, dtype=data.dtype, non_blocking=True)
                        prediction = predictor.predict_sliding_window_return_logits(data[src_seq:src_seq+1], tgt_code)
                        prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnSeq2Seq_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnSeq2SeqDataset(expected_preprocessed_folder, [k],
                                                num_images_properties_loading_threshold=0)
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file = join(output_folder, k + '.npz')

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=8 * dist.get_world_size() if
                                                self.is_ddp else 8)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation PSNR: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(batch_id, next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                self.network_ema.store(self.network.parameters())
                self.network_ema.copy_to(self.network.parameters())
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.network_ema.restore(self.network.parameters())
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()