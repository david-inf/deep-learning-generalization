# Deep-learning-generalization

A work on generalization in deep learning

> Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. Understanding deep learning requires rethinking generalization International Conference on Learning Representations (ICLR), 2017. [arXiv:1611.03530].

### Details

Supervised learning on CIFAR10

<details>
<summary>Models</summary>

- **Net** (simple conv net) `models/simple_mlp.py`:
  - 17,000 params (too few params to reach zero-loss)
  - `learning_rate=0.01`
- **MLP1** `models/simple_mlp.py`:
  - 1,200,000 params
  - `learning_rate=0.01`
- **MLP3** `models/simple_mlp.py`:
  - 1,700,000 params
  - `learning_rate=0.01`
- **AlexNetSmall** `models/alexnet.py`:
  - 1,560,000 params
  - `learning_rate=0.01`
- **InceptionSmall** `models/inception.py`:
  - 1,600,000 params
  - `learning_rate=0.1`
  - Comes with `bn=True`
- **InceptionSmallWithoutBN** `models/inception.py`
  - For ablation studies on batch norm
  - 1,600,000 params
  - `learning_rate=0.1`
  - `bn=False`

</details>

<details>
<summary>Optimization</summary>

- Optimizer: SGD, `momentum=0.9`, decay factor `gamma=0.95` per epoch
- Loaders: `batch_size=128`

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

- **True labels**: original CIFAR10 dataset `p=0.0`
- **Random labels**: dataset with random labels both train and test, probability (fraction) specified by `p=1.0`
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
python main.py --config experiments/AlexNet/AlexNet_0.0_none.yaml --epochs 20
```

```
001: 100%|███████████████████████| 375/375 [00:04<00:00, 88.56batch/s, acc=43.7, loss=1.54]
002: 100%|███████████████████████| 375/375 [00:03<00:00, 102.11batch/s, acc=55.2, loss=1.25]
003: 100%|███████████████████████| 375/375 [00:03<00:00, 99.26batch/s, acc=61.1, loss=1.12]
...
```

Resuming can be done smoothly since the checkpoint path is added to the yaml file and the comet_ml experiment key too

```
python main.py --config experiments/AlexNet/AlexNet_0.0_none.yaml --epochs 40
```

```
...
027: 100%|███████████████████████| 375/375 [00:03<00:00, 95.99batch/s, acc=100, loss=0.00829]
INFO     Zero-loss condition reached at epoch 27 after 107.26s
Test: 100%|██████████████████████| 94/94 [00:00<00:00, 167.94batch/s]
INFO     Test accuracy: 70.2%
028: 100%|███████████████████████| 375/375 [00:03<00:00, 96.36batch/s, acc=99.9, loss=0.00721]
...
```

## Experiments for figure 2

### Examples

## Other results

**Description** | **Result**
--------------- | -----------
The importance of seed with randomization experiments: when no seed is provided, for example in case of resuming (two times here), the model confronts with new data, so it is like testing, except that the new data has an unknown distribution. When the seed is provided, the training continues smoothly in case of resuming. | ![](src/plots/figures/seed_noseed.jpeg)
