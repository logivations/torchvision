import datetime
import json
import os
import os.path
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from PIL import Image
from regression_dataset import ImageRegressionFolder
from sampler import RASampler
from torch import nn
from torch.autograd import Function
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

# from .vision import VisionDataset
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class ConfidenceLossFunction(Function):
    @staticmethod
    def forward(ctx, pred, target):
        """
                This method in the ConfidenceLossFunction uses the logic of the standard MSE loss, but calculates it separately for loaded and confidence outputs. Also, this function does not take the average error value at this stage, but stores the error value for each image separately.
        Then we scale the error from the loaded output to the ground truth value of confidence. Next, we take the average value of loaded loss and confidence loss and sum them.
                :param ctx:
                :param pred:
                :param target:
                :return:
        """
        # Save tensors
        loaded_pred, confidence_pred = pred[:, 0].unsqueeze(1), pred[:, 1].unsqueeze(1)

        loaded_target, confidence_target = target[:, 0].unsqueeze(1), target[
            :, 1
        ].unsqueeze(1)
        ctx.save_for_backward(
            loaded_pred, confidence_pred, loaded_target, confidence_target
        )

        # Calculate loss
        loaded_loss = (loaded_pred - loaded_target) ** 2
        confidence_loss = (confidence_pred - confidence_target) ** 2

        # Scale the loaded loss by the confidence
        scaled_loaded_loss = loaded_loss * confidence_target

        total_loss = scaled_loaded_loss.mean() + confidence_loss.mean()
        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        This method in the ConfidenceLossFunction class computes the gradients of the loss with respect to the predicted values during backpropagation. It first retrieves the predictions and target labels saved during the forward pass, then calculates the gradients for two components: the loaded predictions and the confidence predictions. These gradients are normalized by the batch size, concatenated to form a single gradient for predictions, scaled by grad_output, and returned for further parameter updates in the model.
        :param ctx:
        :param grad_output:
        :return:
        """
        (
            loaded_pred,
            confidence_pred,
            loaded_target,
            confidence_target,
        ) = ctx.saved_tensors
        # print("HERE HERE HERE")
        # Compute gradients for the loaded
        grad_loaded_pred = 2 * (loaded_pred - loaded_target) * confidence_target
        grad_loaded_pred /= loaded_pred.size(0)  # Normalize by batch size

        # Compute gradients for the confidence
        grad_confidence_pred = 2 * (confidence_pred - confidence_target)
        grad_confidence_pred /= confidence_pred.size(0)  # Normalize by batch size

        # Ensure gradients have the same shape as the input pred
        grad_loaded_pred = grad_loaded_pred.expand_as(loaded_pred)
        grad_confidence_pred = grad_confidence_pred.expand_as(confidence_pred)

        grad_pred = torch.cat([grad_loaded_pred, grad_confidence_pred], dim=1)
        grad_pred *= grad_output

        return grad_pred, None


class ConfidenceLoss(nn.Module):
    def __init__(self):
        """"""
        super(ConfidenceLoss, self).__init__()

    def forward(self, pred, target):
        """Apply ConfidenceLoss function"""
        return ConfidenceLossFunction.apply(pred, target)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, annotations_file, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)

        self.image_files = [
            f
            for f in os.listdir(self.root_dir)
            if f.lower() in IMG_EXTENSIONS and f in self.annotations
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert("RGB")

        if "confidence" not in self.annotations[img_name] or "loaded" not in self.annotations[img_name]:
            raise KeyError(f"'confidence' key is missing for image {img_name}")


        loaded = self.annotations[img_name]["loaded"]
        confidence = self.annotations[img_name]["confidence"]

        target = torch.tensor([loaded, confidence], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    criterion = ConfidenceLoss().to(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("loaded_mse", utils.SmoothedValue(window_size=10, fmt="{value:.4f}"))
    metric_logger.add_meter("conf_mse", utils.SmoothedValue(window_size=10, fmt="{value:.4f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=10, fmt="{value:.4f}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)  # output shape: [batch_size, 2]
            loss = criterion(output, target)
            loaded_loss = nn.MSELoss()(output[:, 0], target[:, 0])
            conf_loss = nn.MSELoss()(output[:, 1], target[:, 1])

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["loaded_mse"].update(loaded_loss.item(), n=batch_size)
        metric_logger.meters["conf_mse"].update(conf_loss.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def r2_score_torch(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


import numpy as np

ACCURACY_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.3


def threshold_accuracy(gt: list, pred: np.ndarray, threshold: float = ACCURACY_THRESHOLD) -> bool:
    """
    an image is accurate, if
        1) the predicted confidence is correct, by up to 0.3 error
        2) if the ground truth confidence is > 0.8, also the predicted load state must be correct by up to 0.3 error
    :param gt: Annotation value
    :param pred: Model outputs
    :param threshold: Permitted error
    :return:
    """
    if pred[1] > gt[1] - threshold and pred[1] < gt[1] + threshold:
        if gt[1] >= CONFIDENCE_THRESHOLD:
            if pred[0] > gt[0] - threshold and pred[0] < gt[0] + threshold:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    all_outputs = []
    all_targets = []
    correct = 0
    threshold = 0.3

    with torch.inference_mode():
        running_loss = 0
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            # print("Criteria", criterion)
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())

            loaded_mse = nn.MSELoss()(output[:, 0], target[:, 0])
            conf_mse = nn.MSELoss()(output[:, 1], target[:, 1])
            for pred, true in zip(output.cpu().numpy(), target.cpu().numpy()):
                if threshold_accuracy(true, pred, threshold):
                    correct += 1

            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["loaded_mse"].update(loaded_mse.item(), n=batch_size)
            metric_logger.meters["conf_mse"].update(conf_mse.item(), n=batch_size)
            metric_logger.meters["loss"].update(loss.item(), n=batch_size)
            running_loss += loss.item() * image.size(0)
            num_processed_samples += batch_size

    epoch_loss = running_loss / len(data_loader)
    print("EPOCH LOSS: ", epoch_loss, len(data_loader))
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    loaded_r2 = r2_score_torch(all_outputs[:, 0], all_targets[:, 0])
    conf_r2 = r2_score_torch(all_outputs[:, 1], all_targets[:, 1])
    accuracy = correct / len(data_loader.dataset)
    r2 = (loaded_r2 * all_targets[:, 1]).mean()
    metric_logger.synchronize_between_processes()

    print(
        f"{header} Loaded MSE: {metric_logger.loaded_mse.global_avg:.4f} | "
        f"Loaded R²: {loaded_r2:.4f}\n"
        f"Confidence MSE: {metric_logger.conf_mse.global_avg:.4f} | "
        f"Confidence R²: {conf_r2:.4f} |"
        f"Accuracy@{threshold}: {accuracy:.4f} | "
        f"Custom loss: {metric_logger.loss} |"
        f"R2: {r2} |"
    )

    return {
        "loaded_mse": metric_logger.loaded_mse.global_avg,
        "conf_mse": metric_logger.conf_mse.global_avg,
        "loaded_r2": loaded_r2,
        "conf_r2": conf_r2,
        "loss": metric_logger.loss,
        f"acc@{threshold}": accuracy,
    }


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, testdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    # Load training data
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_train from {cache_path}")
        dataset_train, _ = torch.load(cache_path, weights_only=False)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset_train = ImageRegressionFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
            annotations_file=args.annotations_file
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_train, traindir), cache_path)
    print("Train dataset size:", len(dataset_train))
    print("Time taken:", time.time() - st)

    # Load validation data
    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_val from {cache_path}")
        dataset_val, _ = torch.load(cache_path, weights_only=False)
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            backend=args.backend,
            use_v2=args.use_v2,
        )
        dataset_val = ImageRegressionFolder(
            valdir,
            preprocessing,
            annotations_file=args.annotations_file
        )
        if args.cache_dataset:
            print(f"Saving dataset_val to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_val, valdir), cache_path)
    print("Validation dataset size:", len(dataset_val))

    print("Loading test data")
    cache_path = _get_cache_path(testdir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            backend=args.backend,
            use_v2=args.use_v2,
        )
        dataset_test = ImageRegressionFolder(
            testdir,
            preprocessing,
            annotations_file=args.annotations_file
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, testdir), cache_path)
    print("Test dataset size:", len(dataset_test))
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    test_dir = os.path.join(args.data_path, "test")
    dataset_train, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler = \
        load_data(train_dir, val_dir, test_dir, args)

    collate_fn = default_collate

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2)
    )
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = ConfidenceLoss().to(device)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        val_metrics = evaluate(model, criterion, data_loader_val, device=device, log_suffix="Validation")
        if model_ema:
            evaluate(model_ema, criterion, data_loader_val, device=device, log_suffix="Validation EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            print("Validation Accuracy", val_metrics[f"acc@{0.3}"])
            if val_metrics[f"acc@{0.3}"] > 0.880:
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default="DEFAULT", type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the JSON annotations file")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
