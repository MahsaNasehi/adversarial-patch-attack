# MNIST Adversarial Patch

Modular implementation to train an adversarial patch on MNIST forcing predictions to a target class.

Examples:

1. Train model:

```bash
python -m src.train_model --epochs 3 --save_path mnist_cnn.pth
```

2. Train patch (requires trained model):

```bash
python -m src.train_patch --model_path mnist_cnn.pth --epochs 10 --patch_size 7 --save_path mnist_patch.pth
```

3. Evaluate and visualize:

```bash
python -m src.eval_patch --model_path mnist_cnn.pth --patch_path mnist_patch.pth --out_dir ./out
```

Run tests:

```bash
pytest -q
```
