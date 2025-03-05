
import os
import time
from utils import N
import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ExponentialLR

import wandb


def save_checkpoint(opts, model, optimizer, epoch, loss):
    """ Save a model checkpoint so training can be resumed and also wandb logging """
    fname = os.path.join(
        opts.checkpoint_dir,
        f"e_{epoch:03d}_{opts.model_name}_{opts.run_name}.pt"
    )
    info = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        loss=loss
    )
    torch.save(info, fname)
    wandb.save(fname)
    print(f"Saved checkpoint {fname}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """ Load a model checkpoint to resume training """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # load from given checkpoint path
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # load weights and optimizer in those given
    # this means that the inizialized model and optimizer are updated
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)  # last completed epoch
    loss = checkpoint.get("loss", float("inf"))  # loss at checkpoint

    print(f"Resuming from epoch {epoch}")
    return epoch, loss


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


def train_loop(opts, model, optimizer, train_loader, val_loader):
    """ Training loop """

    criterion = torch.nn.CrossEntropyLoss()  # expects logits
    scheduler = ExponentialLR(optimizer, gamma=opts.lr_decay)

    wandb.watch(model, criterion, log="all", log_freq=opts.log_every, log_graph=False)

    _start = time.time()
    step = 0  # logging step
    for epoch in range(1, opts.num_epochs + 1):
        losses, accs = [], []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (X, y) in enumerate(tepoch):
                model.train()
                tepoch.set_description(f"Epoch {epoch}")

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
                loss.backward()  # backprop
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
                    # log to wandb
                    wandb.log({
                        "epoch": epoch,
                        "train loss": train_loss,
                        "train acc": train_acc
                    }, step=step)
                    # log to console
                    tepoch.set_postfix(loss=train_loss, acc=100.*train_acc)
                    tepoch.update()
                    step += 1

                    # Check zero-loss condition
                    if train_loss < 1e-2:
                        # time to reach interpolation treshold
                        # this will be updated at each zero-loss condition
                        # TODO: here or after training is ended but still at zero-loss?
                        # TODO: otherwise I can compute test error at best model?
                        _zero_loss_time = time.time() - _start  # seconds
                        print(f"Zero-loss condition reached at epoch {epoch}")
                        print(f"Time to zero-loss: {_zero_loss_time:.2f} seconds")
                        # test error at interpolation treshold
                        test_acc = test(opts, model, val_loader, "Test")
                        print(f"Test accuracy: {100.*test_acc:.1f}%")
                        wandb.log({
                            "test acc": test_acc,
                            "test error": 1. - test_acc,
                            "time to overfit": _zero_loss_time
                        })
        
        scheduler.step()  # update learning rate

        if epoch % opts.checkpoint_every == 0 or epoch == opts.num_epochs:
            # save every checkpoint_every epochs and at the end
            save_checkpoint(opts, model, optimizer, epoch, loss)

    print(f"Training completed in {time.time() - _start:.2f} seconds")
    # save final model
    # save_checkpoint(opts, model, optimizer, epoch, loss)
    # test error at best model
    # test_acc = test(opts, model, val_loader, "Final Test")
    # print(f"Final test accuracy: {100.*test_acc:.1f}%")
    # wandb.log({
    #     "final test acc": test_acc,
    #     "final test error": 1. - test_acc
    # })
