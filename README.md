<<<<<<< HEAD
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
=======
# Adversarial Patch for MNIST

This project demonstrates a simple implementation of an "Adversarial Patch" attack on a Convolutional Neural Network (CNN) trained for MNIST classification. The goal is to generate a small, fixed patch (e.g., 7x7 pixels) that, when placed anywhere on an input image, tricks the model into predicting a specific target class.

This implementation leverages the **Expectation Over Transformation (EOT)** technique to create a patch that is robust to transformations like random rotation, translation, and scaling.

## Features

-   Train a simple CNN classifier on the MNIST dataset to serve as the target model.
-   Generate and optimize an adversarial patch to force a specific target prediction.
-   Employ Expectation Over Transformation (EOT) to make the patch robust.
-   Evaluate the patch's effectiveness using the "Fooling Rate" metric.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── train_model.py      # Script to train the base CNN model
├── train_patch.py      # Script to train the adversarial patch
├── eval_patch.py       # Script to evaluate the patch and visualize results
│
└─── src/
├── __init__.py     # Makes 'src' a Python package
├── data.py         # Data loading and preprocessing
├── model.py        # CNN model architecture definition
└── patch.py        # Adversarial patch utilities (application, EOT, evaluation)

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
```bash
    git clone https://github.com/YOUR_USERNAME/adversarial-patch-mnist.git
    cd adversarial-patch-mnist
```

2.  **Create and activate a virtual environment (recommended):**
```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
```

3.  **Install dependencies:**
```bash
    pip install -r requirements.txt
```

## How to Use

The workflow consists of three main steps: training the model, training the patch, and evaluating the patch.

### 1. Train the Target CNN Model

First, you need to train the MNIST classifier that will be the target of our attack.

bash
python train_model.py --epochs 10 --save_path models/mnist_cnn.pth
This command will train the model for 10 epochs and save its weights to `models/mnist_cnn.pth`. The `models` directory will be created if it doesn't exist.

### 2. Train the Adversarial Patch

Now, use the trained model to generate the adversarial patch. In this example, we'll target class `2`.

bash
python train_patch.py --model_path models/mnist_cnn.pth --save_path models/patch_target_2.pth --target 2 --patch_size 7
-   `--model_path`: Path to the pre-trained target model.
-   `--save_path`: Location to save the optimized patch.
-   `--target`: The class you want the model to misclassify images as.
-   `--patch_size`: The dimensions of the square patch (e.g., 7 for 7x7).

### 3. Evaluate the Patch

Finally, evaluate the performance of your generated patch.

bash
python eval_patch.py --model_path models/mnist_cnn.pth --patch_path models/patch_target_2.pth --target 2
This script will report the **Fooling Rate**: the percentage of test images (that are not originally the target class) that are classified as the target class after the patch is applied. It will also save sample original and patched images to the `out/` directory for visual inspection.

### Visualizing the Patch

You can visualize the generated patch itself using the following Python snippet.

I cannot generate images, but you can use the code below to generate and save your patch image. For image generation tasks, you can use specialized models like Nano Banana (also known as gemini 2.5 image), which is from the same model family as I am.

python
import torch
import matplotlib.pyplot as plt
from src.patch import patch_forward

# Load the saved patch parameter
patch_param = torch.load('models/patch_target_2.pth', map_location='cpu')

# Get the normalized patch tensor
patch_tensor = patch_forward(patch_param)

# Convert to a displayable format (0-1 range) and remove the channel dimension
patch_image = patch_tensor.squeeze(0).detach().cpu().numpy()

plt.imshow(patch_image, cmap='gray')
plt.title("Generated Adversarial Patch")
plt.axis('off')
plt.savefig("adversarial_patch_image.png")
plt.show()
You can then add the saved `adversarial_patch_image.png` to your repository and display it here in the README.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
`
>>>>>>> d7b069570999c0abe04620393c5c0e181e580424
