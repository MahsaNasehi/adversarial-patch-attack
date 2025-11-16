# # src/patch.py
# """Adversarial patch utilities: parameterization, placement and evaluation."""
# import torch
# import torch.nn as nn
# import numpy as np
# from typing import Tuple, List

# from .model import MNIST_MEAN, MNIST_STD

# def patch_forward(patch_param: torch.Tensor) -> torch.Tensor:
#     """Map unconstrained patch_param to normalized image space.

#     Args:
#         patch_param: Tensor of shape (1, H, W) with unconstrained values (requires_grad=True)

#     Returns:
#         Tensor of shape (1, H, W) in normalized MNIST space ((x-mean)/std)
#     """
#     p = torch.tanh(patch_param)  # (-1,1)
#     p = (p + 1.0) / 2.0  # (0,1)
#     p = (p - MNIST_MEAN) / MNIST_STD
#     return p

# def place_patch(images: torch.Tensor, patch_tensor: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, List[Tuple[int,int]]]:
#     """Place the patch at random positions for each image in the batch.

#     Args:
#         images: (B,1,28,28)
#         patch_tensor: (1, ph, pw) already normalized
#         patch_size: int patch spatial size (square)

#     Returns:
#         patched_images: new tensor with patch applied (copy)
#         positions: list of (h_off, w_off) for each image
#     """
#     B = images.shape[0]
#     patched = images.clone()
#     H, W = 28, 28
#     positions = []
#     # ensure patch_tensor shape
#     for i in range(B):
#         h_off = np.random.randint(0, H - patch_size + 1)
#         w_off = np.random.randint(0, W - patch_size + 1)
#         patched[i, :, h_off:h_off+patch_size, w_off:w_off+patch_size] = patch_tensor
#         positions.append((h_off, w_off))
#     return patched, positions

# @torch.no_grad()
# def eval_patch(model: nn.Module, patch_param: torch.Tensor, dataloader, target_class: int = 2, patch_size: int = 7, max_batches: int = 100) -> float:
#     """
#     Evaluate the fraction of non-target examples that become predicted as target_class after applying patch.
#     Returns fooling rate in [0,1].
#     """
#     device = next(model.parameters()).device
#     model.eval()
#     correct = 0
#     total = 0
#     batches = 0
#     p = patch_forward(patch_param).to(device)
#     for x,y in dataloader:
#         x = x.to(device)
#         y = y.to(device)
#         patched, _ = place_patch(x, p, patch_size)
#         logits = model(patched)
#         preds = logits.argmax(dim=1)
#         mask = (y != target_class)
#         if mask.sum().item() > 0:
#             correct += (preds[mask] == target_class).sum().item()
#             total += mask.sum().item()
#         batches += 1
#         if batches >= max_batches:
#             break
#     return 0.0 if total == 0 else correct / total

# src/patch.py
"""
Adversarial patch utilities with simple EOT transforms for MNIST.
- patch_forward: map param -> normalized patch
- place_patch_random: place patch at random position (no transform)
- apply_eot_transforms: apply random affine/noise transforms to a batch
- place_patch_and_transform: convenience to place then transform (used in training)
- eval_patch: evaluate fooling rate using multiple EOT samples
"""
import torch
import numpy as np
from typing import Tuple, List
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .model import MNIST_MEAN, MNIST_STD

def patch_forward(patch_param: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained patch_param to normalized image space.

    Args:
        patch_param: Tensor (1, ph, pw) with unconstrained values
    Returns:
        Tensor (1, ph, pw) in normalized MNIST space ((x-mean)/std)
    """
    p = torch.tanh(patch_param)  # (-1,1)
    p = (p + 1.0) / 2.0          # (0,1)
    p = (p - MNIST_MEAN) / MNIST_STD
    return p

def place_patch_random(images: torch.Tensor, patch_tensor: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, List[Tuple[int,int]]]:
    """
    Place the patch at random positions for each image in the batch (no transform).
    images: (B,1,28,28)
    patch_tensor: (1, ph, pw) already normalized
    Returns: patched_images, positions
    """
    B = images.shape[0]
    patched = images.clone()
    H, W = 28, 28
    positions = []
    for i in range(B):
        h_off = np.random.randint(0, H - patch_size + 1)
        w_off = np.random.randint(0, W - patch_size + 1)
        patched[i, :, h_off:h_off+patch_size, w_off:w_off+patch_size] = patch_tensor
        positions.append((h_off, w_off))
    return patched, positions

def apply_eot_transforms(batch: torch.Tensor, angle_range=(-30,30), translate_frac=0.1, scale_range=(0.9,1.1), add_noise_std=0.0):
    """
    Apply random affine transforms + optional noise to a batch of images.
    batch: (B,1,H,W) tensor in normalized space (same as model input)
    Returns a new tensor of same shape.
    Notes:
      - translate_frac is fraction of image size to sample translate from
      - angles in degrees
      - scale is multiplicative
    """
    B, C, H, W = batch.shape
    out = torch.zeros_like(batch)
    for i in range(B):
        img = batch[i]
        # random params
        angle = float(np.random.uniform(angle_range[0], angle_range[1]))
        max_tx = int(W * translate_frac)
        max_ty = int(H * translate_frac)
        tx = int(np.random.randint(-max_tx, max_tx+1))
        ty = int(np.random.randint(-max_ty, max_ty+1))
        scale = float(np.random.uniform(scale_range[0], scale_range[1]))
        # torchvision.transforms.functional.affine expects tensor (C,H,W)
        # shear=0
        img_t = TF.affine(
            img,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=[0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )        
        if add_noise_std and add_noise_std > 0.0:
            noise = torch.randn_like(img_t) * add_noise_std
            img_t = img_t + noise
        out[i] = img_t
    return out

def place_patch_and_transform(images: torch.Tensor, patch_tensor: torch.Tensor, patch_size: int,
                              angle_range=(-30,30), translate_frac=0.1, scale_range=(0.9,1.1), add_noise_std=0.0):
    """
    Convenience: place patch randomly, then apply EOT transform to whole image batch.
    Returns transformed patched images and positions (before transform).
    """
    patched, positions = place_patch_random(images, patch_tensor, patch_size)
    transformed = apply_eot_transforms(patched, angle_range=angle_range,
                                       translate_frac=translate_frac, scale_range=scale_range,
                                       add_noise_std=add_noise_std)
    return transformed, positions

@torch.no_grad()
def eval_patch(model: torch.nn.Module, patch_param: torch.Tensor, dataloader, target_class: int = 2,
               patch_size: int = 7, eot_samples: int = 10, device='cpu'):
    """
    Evaluate fooling rate by sampling eot_samples transforms per batch and averaging.
    Returns fraction of *non-target* images where the model predicts target_class
    averaged over transforms.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    batches = 0
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        B = x.shape[0]
        # compute p once
        p = patch_forward(patch_param).to(device)
        # accumulate predictions across samples
        preds_sum = torch.zeros(B, dtype=torch.int32, device=device)
        counts = 0
        for _ in range(eot_samples):
            patched, _ = place_patch_and_transform(x, p, patch_size)
            logits = model(patched)
            preds = logits.argmax(dim=1)
            # count predictions equal to target_class
            preds_sum += (preds == target_class).to(torch.int32)
            counts += 1
        # fraction per image
        frac = preds_sum.float() / float(counts)
        mask = (y != target_class)
        if mask.sum().item() > 0:
            # use fraction threshold 0.5 to declare 'fooling' (majority of transforms fooled)
            fooled = (frac[mask] >= 0.5).sum().item()
            correct += fooled
            total += mask.sum().item()
        batches += 1
        if batches >= 100:
            break
    return 0.0 if total == 0 else correct / total
