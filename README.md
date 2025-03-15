# Deep-learning-generalization

A work on generalization in deep learning

### Details

Supervised learning on CIFAR10

- Models:
    - MLP
    - (Simple-Net)
    - AlexNet
    - Inception-V3
- Optimization
    - Optimizer: SGD

## Experiments

Experiments for figure 1
- Loss per training step varying randomization test
    - True labels: original dataset
    - Random labels: full dataset with random labels
    - Shuffled pixels: fixed permutation for all images
    - Random pixels: different permutation ofr each image
    - Gaussian: random pixels according to gaussian with mean and std matching the dataset
- Time to overfit (time to reach interpolation threshold) againts label corruption for each network
- Test error (at interpolaton threshold) against label corruption for each network

| Experiment | Results |
| ---------- | ------- |
| Learning curves. Fixed net with varying randomization in data. | plot |
| Convergence slowdown. Time to overfit at different label corruption levels. | plot |
| Generalization error growth. Same as the previous but we look for the test error. | plot |

Experiments for figure 2
- 

## References
