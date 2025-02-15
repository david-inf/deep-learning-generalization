
from utils import N
import numpy as np
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import ExponentialLR


def test(opts, model, test_loader, msg="Test"):
    model.eval()

    correct = []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for (X, y) in test_loader:
                tepoch.set_description(msg)

                X, y = X.to(opts.device), y.to(opts.device)
                out = model(X)  # logits: [N, K]
                pred = np.argmax(N(out), axis=1)  # array of ints
                label = np.argmax(N(y), axis=1)  # {0,...,9}
                c = list(pred == label)  # [0,1,0,0,0,1,1...]
                correct.extend(c)

    return np.mean(correct)


def train_loop(opts, model, optimizer, train_loader):

    criterion = torch.nn.CrossEntropyLoss()  # expects logits

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
                y = y.to(opts.device)  # [N, K]  one-hot
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
                    # TODO: validation
                    # val_acc = test(opts, model, val_loader, "Validation")
                    tepoch.set_postfix(loss=train_loss, acc=100.*train_acc)
                    tepoch.update()
                    step += 1
