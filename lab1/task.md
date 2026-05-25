# Lab 1 — Introduction to PyTorch

## Task

Get acquainted with the PyTorch framework and complete three tasks:

1. **Regression** via the universal approximation theorem — manual differentiation
2. **Binary classification** using PyTorch autograd
3. **Multi-class image classification** — train a fully-connected MLP on 3 CIFAR-100 classes and improve test accuracy

The lab runs on Google Colab (badge at the top of `notebook.ipynb`) or locally.

---

## Part 3 — Variant (CIFAR-100 class selection)

| Position | Rule |
|----------|------|
| Class 1 | Group number + 7 |
| Class 2 | Variant number + 30 |
| Class 3 | ИУ5: variant + 50 / ГУИМЦ: 90 / International: 93 |

Set your classes in `configs/config.yaml` → `classes: [c1, c2, c3]`.

---

## Self-study Tasks (Part 3)

1. Analyze results. What do train/test accuracy reveal? Which classes perform best?
2. Does overfitting occur? How to mitigate it without regularization?
3. Double the batch size while keeping total iterations constant. What changes?
4. Reduce lr by 3×, increase epochs by 3×. Does accuracy improve?
5. Change hidden layer size and neuron count. Find the best hyperparameters.
6. List and explain what helped improve accuracy.

**Note:** When changing batch size or lr, adjust epochs proportionally to keep the total number of gradient steps constant. This isolates the effect of each hyperparameter.

---

## Results Table (to fill in)

| Model config | `lr` | `batch_size` | `epochs` | Train Acc | Test Acc | Comment |
|---|---|---|---|---|---|---|
| FC(10), FC(3) | 0.005 | 128 | 250 | 99.53% | 74.67% | Baseline |
| FC(10), FC(3) | 0.005 | 256 | 500 | | | 2× batch, 2× epochs |
| FC(10), FC(3) | 0.0017 | 128 | 750 | | | lr÷3, epochs×3 |
| FC(X), FC(3) | | | | | | Modified architecture |

---

## Defense Questions

1. Fully-connected neural network — structure, computations, layer purposes.
2. Number of neurons, connections, and weights in an FC network.
3. Regression vs. classification tasks — which loss functions apply?
4. Dataset structure — purpose of train/val/test splits.
5. SGD algorithm — hyperparameters, difference from batch GD.
6. Epoch, iteration, batch — definitions and relationships.
7. Supervised / unsupervised / reinforcement learning — examples of each.
8. Draw architecture diagrams for all three networks in the lab.
