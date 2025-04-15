
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.backends.cudnn as cudnn

from utils import N, LOG, update_yaml, AverageMeter


class TrainState:
    """Automate checkpointing"""

    def __init__(self, model, optimizer, scheduler, epoch, step, runtime):
        self.model = model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.scheduler = scheduler.state_dict()
        self.epoch = epoch  # last epoch reached
        self.step = step  # last step reached
        self.runtime = runtime  # total runtime

    def update(self, model, optimizer, scheduler, epoch, step, runtime):
        self.model = model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.scheduler = scheduler.state_dict()
        self.epoch = epoch
        self.step = step
        self.runtime = runtime

    def __dict__(self):
        """Use for saving checkpoints"""
        return {"model_state_dict": self.model,
                "optimizer_state_dict": self.optimizer,
                "scheduler_state_dict": self.scheduler, "epoch": self.epoch,
                "step": self.step, "runtime": self.runtime}


def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler):
    """Load a model checkpoint to resume training"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")

    # load from given checkpoint path
    LOG.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # load weights and optimizer in those given
    # this means that the initialized model and optimizer are updated
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)  # last completed epoch
    step = checkpoint.get("step", 0)  # last logged step
    runtime = checkpoint.get("runtime", 0.)

    # print(f"Resuming from epoch {epoch}, step {step}")
    return epoch, step, runtime


def test(opts, model, loader):
    """Evaluate model on test/validation set"""
    losses, accs = AverageMeter(), AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()  # scalar value
    model.eval()
    with torch.no_grad():
        for (X, y) in loader:
            # Get data and forward pass
            X, y = X.to(opts.device), y.to(opts.device)
            out = model(X)  # logits: [N, K]
            # Compute loss
            loss = criterion(out, y)
            losses.update(N(loss), X.size(0))
            # Compute accuracy
            acc = np.mean(np.argmax(N(out), axis=1) == N(y))
            accs.update(acc, X.size(0))

    return losses.avg, accs.avg


def save_checkpoint(trainer: TrainState, opts, fname=None):
    """Save a model checkpoint to be resumed later"""
    info = trainer.__dict__()
    if not fname:
        fname = f"e_{info["epoch"]:03d}_{opts.experiment_name}.pt"

    # save various stuffs from Trainer class
    output_dir = os.path.join(opts.checkpoint_dir, fname)
    torch.save(info, output_dir)

    # Update yaml file with checkpoint name
    update_yaml(opts, "resume_checkpoint", output_dir)
    LOG.info(f"Saved checkpoint {fname} at epoch {info["epoch"]}, "
             f"step {info["step"]}, runtime {info["runtime"]:.2f}s")


def train_loop(opts, model, train_loader, val_loader, experiment, resume_from=None):
    """Train loop for Figure 1 experiments"""
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()  # expects logits and labels
    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate,
                          momentum=opts.momentum, weight_decay=opts.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    start_time, prev_runtime = time.time(), 0.  # previous duration in case of resuming

    if resume_from:
        # load checkpoint
        last_epoch, last_step, prev_runtime = load_checkpoint(
            resume_from, model, optimizer, scheduler)
        start_epoch += last_epoch
        step += last_step
        LOG.info(f"Resuming training from epoch {start_epoch}, step {step},"
                 f" previous runtime {prev_runtime:.2f}s")

    # save training objects and info for checkpointing
    trainer = TrainState(model, optimizer, scheduler,
                         start_epoch, step, prev_runtime)

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)

        if opts.figure1:
            # check if zero-loss has been reached
            if opts.interp_reached and not opts.curve:
                ckp_runtime = prev_runtime + time.time() - start_time  # add duration of this run
                trainer.update(model, optimizer, scheduler,
                               epoch, step, ckp_runtime)
                fname = f"e_{trainer.epoch:03d}_{opts.experiment_name}_interp.pt"
                save_checkpoint(trainer, opts, fname)
                LOG.info("Interpolation threshold reached, "
                         "and no need to continue, breaking training...")
                break

        step, train_loss, train_acc = train_epoch(
            opts, model, train_loader, val_loader, experiment, criterion,
            optimizer, step, epoch)

        if opts.figure1:
            # Check zero-loss condition at each epoch
            check_interp(opts, model, val_loader, experiment, start_time,
                         prev_runtime, epoch, train_loss, train_acc)

        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs and at the end
            ckp_runtime = prev_runtime + time.time() - start_time  # add duration of this run
            trainer.update(model, optimizer, scheduler,
                           epoch, step, ckp_runtime)
            save_checkpoint(trainer, opts)

    # add this run duration to the previous one
    runtime = time.time() - start_time
    prev_runtime += runtime
    LOG.info(f"Training completed in {runtime:.2f}s <> "
             f"Current runtime: {prev_runtime:.2f}s")
    experiment.log_metric("runtime", prev_runtime)
    LOG.info(f"Current training at epoch {epoch}, step {step}")


def train_epoch(opts, model, train_loader, val_loader, experiment, criterion, optimizer, step, epoch):
    """Train over a single epoch"""
    losses, accs = AverageMeter(), AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, y) in enumerate(tepoch):
            model.train()
            tepoch.set_description(f"{epoch:03d}")

            # -----
            # move data to device
            X = X.to(opts.device)  # [N, C, W, H]
            y = y.to(opts.device)  # [N]
            # forward pass
            output = model(X)  # logits: [N, K]
            loss = criterion(output, y)  # scalar value
            # metrics
            losses.update(N(loss), X.size(0))  # add loss for current batch
            acc = np.mean(np.argmax(N(output), axis=1) == N(y))
            accs.update(acc, X.size(0))  # add accuracy for current batch
            # backward pass
            optimizer.zero_grad()
            loss.backward()   # backprop
            optimizer.step()  # update model
            # -----

            if batch_idx % opts.log_every == 0 or batch_idx == len(train_loader) - 1:
                # Compute training metrics and log to comet_ml
                train_loss, train_acc = losses.avg, accs.avg
                experiment.log_metrics({"loss": train_loss, "acc": train_acc},
                                       step=step)
                if opts.figure1:
                    # Just log to console
                    tepoch.set_postfix(train_loss=train_loss,
                                       train_acc=train_acc)
                else:
                    # Compute validation metrics too and log to comet_ml
                    with experiment.test():
                        val_loss, val_acc = test(opts, model, val_loader)
                        experiment.log_metrics({"loss": val_loss, "acc": val_acc},
                                               step=step)
                        # Log to console
                        tepoch.set_postfix(train_loss=train_loss, train_acc=train_acc,
                                           val_loss=val_loss, val_acc=val_acc)
                tepoch.update()
                step += 1

    return step, train_loss, train_acc


def check_interp(opts, model, test_loader, experiment, start_time, prev_runtime, epoch, train_loss, train_acc):
    """
    Check if model have reached the interpolation threshold
    if so validate over test set
    """
    if not opts.interp_reached:
        # if not yet at interp thresh, do a check
        if train_loss < 1e-2 or train_acc > 0.997:
            # interp thresh reached!
            opts.interp_reached = True
            # update yaml, crucial when resuming
            update_yaml(opts, "interp_reached", True)

            with experiment.test():
                # When interpolation threshold is reached
                # log test error and time to reach interpolation threshold
                # this must be done just one time
                zero_loss_time = time.time() - start_time + prev_runtime  # seconds
                LOG.info(f"Zero-loss condition reached at epoch {epoch}"
                         f" after {zero_loss_time:.2f}s")

                # test error at interpolation treshold
                _, test_acc = test(opts, model, test_loader)
                LOG.info(f"Test accuracy: {100.*test_acc:.1f}%")

                # log to comet_ml
                experiment.log_metrics({"acc": test_acc, "error": 1. - test_acc,
                                        "time_to_overfit": zero_loss_time/60,  # minutes
                                        "epochs_to_overfit": epoch,
                                        "label_corruption": opts.label_corruption_prob})
