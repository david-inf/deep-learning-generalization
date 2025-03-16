
import os
import time
from utils import N
import numpy as np
from tqdm import tqdm

# logging to comet_ml
from comet_ml.integration.pytorch import watch

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from utils import LOG, update_yaml


def save_checkpoint(opts, model, optimizer, scheduler, epoch, step, loss, runtime):
    """ Save a model checkpoint so training can be resumed and also wandb logging """
    fname = os.path.join(
        opts.checkpoint_dir,
        f"e_{epoch:03d}_{opts.experiment_name}.pt"
    )
    info = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        epoch=epoch,  # last epoch
        step=step,  # last step
        loss=loss,  # last computed loss
        runtime=runtime,  # duration of the run
    )
    torch.save(info, fname)
    # TODO: get from comet_ml, no need, one can use log_model
    # Update yaml file with checkpoint name
    update_yaml(opts, "resume_checkpoint", fname)
    LOG.info(f"Saved checkpoint {fname} at epoch {epoch}, step {step}, runtime {runtime:.2f}s")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """ Load a model checkpoint to resume training """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # load from given checkpoint path
    LOG.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # load weights and optimizer in those given
    # this means that the inizialized model and optimizer are updated
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)  # last completed epoch
    step = checkpoint.get("step", 0)  # last logged step
    loss = checkpoint.get("loss", float("inf"))  # loss at checkpoint
    runtime = checkpoint.get("runtime", 0.)

    # print(f"Resuming from epoch {epoch}, step {step}")
    return epoch, step, loss, runtime


def test(opts, model, test_loader, msg="Test"):
    """ Evaluate model on test/validation set """
    model.eval()

    correct = []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for (X, y) in tepoch:
                tepoch.set_description(msg)

                X, y = X.to(opts.device), y.to(opts.device)
                out = model(X)  # logits: [N, K]
                pred = np.argmax(N(out), axis=1)  # array of ints
                label = np.argmax(N(y), axis=1)  # {0,...,9}
                c = list(pred == label)  # [0,1,0,0,0,1,1...]
                correct.extend(c)

    return np.mean(correct)


def train_loop(opts, model, train_loader, val_loader,
               experiment, resume_from=None):
    """
    Training loop with with resuming routine. This accounts for training
    ended before the number of epochs is reached or when one wants
    to train the model further.
    """

    criterion = torch.nn.CrossEntropyLoss()  # expects logits
    optimizer = optim.SGD(
        model.parameters(), lr=opts.learning_rate,
        momentum=opts.momentum, weight_decay=opts.weight_decay
    )
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    start_epoch, step = 1, 0  # last training epoch and step
    _start, prev_runtime = time.time(), 0.  # previous duration in case of resuming
    # TODO: is it possible to account for epoch not ended?
    # the checkpoint is at some epoch
    # but at comet_ml may be some steps more
    if resume_from:
        # load checkpoint
        last_epoch, last_step, _, prev_runtime = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch += last_epoch
        step += last_step
        LOG.info(f"Resuming training from epoch {start_epoch}, step {step}, previous runtime {prev_runtime:.2f}s")

    # if not resume_from:
        # this avoids duplicated graphs
        # watch(model)

    for epoch in range(start_epoch, opts.num_epochs + 1):
        experiment.log_current_epoch(epoch)
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                model.train()
                tepoch.set_description(f"{epoch:03d}")

                ## -----
                # move data to device
                X = X.to(opts.device)  # [N, C, W, H]
                y = y.to(opts.device)  # [N, K] one-hot
                # forward pass
                optimizer.zero_grad()
                out = model(X)  # logits: [N, K]
                loss = criterion(out, y)
                loss = torch.mean(loss)  # scalar value
                # backward pass
                loss.backward()   # backprop
                optimizer.step()  # update model
                # metrics
                losses.append(N(loss))  # add loss for current batch
                pred = np.argmax(N(out), axis=1)  # array of ints
                label = np.argmax(N(y), axis=1)  # {0,...,9}
                acc = np.mean(pred == label)
                accs.append(acc)  # add accuracy for current batch
                ## -----

                if batch_idx % opts.log_every == 0:
                    train_loss = np.mean(losses[-opts.batch_window:])
                    train_acc = np.mean(accs[-opts.batch_window:])
                    # log to comet_ml
                    experiment.log_metrics({
                        "loss": train_loss,
                        "acc": train_acc,
                    }, step=step)
                    # log to console
                    tepoch.set_postfix(loss=train_loss, acc=100.*train_acc)
                    tepoch.update()
                    step += 1

        # Check zero-loss condition at each epoch
        # uses last computed loss and accuracy
        if not opts.interp_reached:
            # TODO: should be better the full epoch loss and accuracy?
            # epoch_loss, epoch_acc = np.mean(losses), np.mean(accs)
            if train_loss < 1e-2 or train_acc > 0.9995:
                opts.interp_reached = True
                # update yaml, crucial when resuming
                update_yaml(opts, "interp_reached", True)
                with experiment.validate():
                    # When interpolation threshold is reached
                    # log test error and time to reach interpolation threshold
                    # this must be done just one time
                    _zero_loss_time = time.time() - _start + prev_runtime  # seconds
                    LOG.info(f"Zero-loss condition reached at epoch {epoch} after {_zero_loss_time:.2f}s")
                    # test error at interpolation treshold
                    test_acc = test(opts, model, val_loader, "Test")
                    LOG.info(f"Test accuracy: {100.*test_acc:.1f}%")
                    # log to comet_ml
                    experiment.log_metrics({
                        "acc": test_acc,
                        "error": 1. - test_acc,
                        "time_to_overfit": _zero_loss_time,
                        "label_corruption": opts.label_corruption_prob
                    })

        scheduler.step()  # update learning rate at each epoch

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs and at the end
            ckp_runtime = prev_runtime + time.time() - _start  # add duration of this run
            save_checkpoint(opts, model, optimizer, scheduler, epoch, step, loss, ckp_runtime)

    prev_runtime += time.time() - _start  # add this run duration to the previous one
    LOG.info(f"Training completed in {prev_runtime:.2f}s")
    LOG.info(f"Total duration {prev_runtime:.2f}s")
    # save final model
    # save_checkpoint(opts, model, optimizer, epoch, loss)
    # test error at best model
    # test_acc = test(opts, model, val_loader, "Final Test")
    # print(f"Final test accuracy: {100.*test_acc:.1f}%")
    # wandb.log({
    #     "final test acc": test_acc,
    #     "final test error": 1. - test_acc
    # })
    experiment.log_metric("runtime", prev_runtime)
