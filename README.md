# Deep-learning-generalization

A work on generalization in deep learning

> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. Understanding deep learning requires rethinking generalization International Conference on Learning Representations (ICLR), 2017. [arXiv:1611.03530].

### Details

Supervised learning on CIFAR10

<details>
<summary>Models</summary>

- **Net** (simple conv net) `models/simple_mlp.py`:
    - 17,000 params (too few params to reach zero-loss)
    - learning_rate=0.01
- **MLP1** `models/simple_mlp.py`:
    - 1,200,000 params
    - learning_rate=0.01
- **MLP3** `models/simple_mlp.py`:
    - 1,700,000 params
    - learning_rate=0.01
- **AlexNetSmall** `models/alexnet.py`:
    - 460,000 params
    - learning_rate=0.01
- **InceptionSmall** `models/inception.py`:
    - 1,600,000 params
    - learning_rate=0.1
</details>
<details>
<summary>Optimization</summary>

- Optimizer: SGD, momentum=0.9, decay factor 0.95 per epoch
- Loaders: batch_size=128
</details>
    

### Code organization

- `main.py` main script from which a single experiment can be launched using command line
- `train.py` used in main script, contains training utilities
- `models/` directory with implemented models (detailed above)
- `cifar10.py` wrapper of torchvision CIFAR10 that supports label and data corruption using `ModifiedCIFAR10` class
- `utils.py` more utilities
- `config.yaml` base configuration for experiments

Use `python main.py --help` to show program arguments

# Experiments

Experiments naming: `model_name`\_`label_corruption_prob`\_`data_corruption_type`

## Experiments for figure 1

No weight decay, dropout or other forms of explicit regularization

<details>
<summary>Learning curves</summary>

Loss per training step varying randomization test
- **True labels**: original CIFAR10 dataset
- **Random labels**: dataset with random labels both train and test, probability (fraction) specified by $p\in(0,1]$
- **Shuffled pixels**: a fixed pixels permutation is applied to train and test images
- **Random pixels**: different pixels permutation for each train and test image
- **Gaussian**: train and test images are generated according to a normal distribution with matching mean and std to the full dataset

Fixed architecture with varying randomization test
</details>

<details>
<summary>Convergence slowdown</summary>

Time to reach the interpolation threshold againts label corruption for each network. One must run 11 experiments for the corrution levels per 3 different architectures

We should see that as the label corruption level increases, the time to reach the interpolation threshold increases as well.
</details>

<details>
<summary>Generalization error growth</summary>

Test error at the interpolaton threshold against label corruption level for each network. Same as the previous experiment, just with another metric
</details>

| **Experiment** | **Results** |
| -------------- | ----------- |
| Learning curves | plot |
| Convergence slowdown | plot |
| Generalization error growth | plot |

### Example

Train `AlexNet` model on original CIFAR10 dataset
```
python main.py --config experiments/AlexNet/AlexNet_0.0_none.yaml
```

```
001: 100%|███████████████████████████████████████████████████████████████| 375/375 [00:04<00:00, 90.48batch/s, acc=42.9, loss=1.56]
002: 100%|███████████████████████████████████████████████████████████████| 375/375 [00:03<00:00, 95.29batch/s, acc=54.1, loss=1.29]
003: 100%|███████████████████████████████████████████████████████████████| 375/375 [00:04<00:00, 93.14batch/s, acc=59.9, loss=1.13]
...
```

One can resume training by providing the checkpoint path in `resume_checkpoint`
```
python main.py --config experiments/AlexNet/AlexNet_0.0_none.yaml --epochs 40
```

```
032: 100%|███████████████████████████████████████████████████████████████| 375/375 [00:04<00:00, 91.29batch/s, acc=100, loss=0.0122]
INFO     Zero-loss condition reached at epoch 32 after 48.84s
Test: 100%|██████████████████████████████████████████████████████████████| 94/94 [00:00<00:00, 165.00batch/s]
INFO     Test accuracy: 88.9%
033: 100%|███████████████████████████████████████████████████████████████| 375/375 [00:04<00:00, 91.53batch/s, acc=99.9, loss=0.013]
...
```

## Experiments for figure 2

### Examples
