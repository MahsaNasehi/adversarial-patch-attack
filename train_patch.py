# # src/train_patch.py
# """
# Script to train an adversarial patch for MNIST with visualization.
# Every few epochs, saves sample images (original vs patched) to track progress.
# """
# import argparse
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from torchvision.utils import save_image

# from src.data import get_dataloaders
# from src.model import SimpleCNN
# from src.patch import patch_forward, place_patch, eval_patch, MNIST_MEAN, MNIST_STD


# def visualize_patch_batch(x, patched, epoch, out_dir, num_images=16):
#     """Save original and patched images for visual inspection."""
#     x_unn = x * MNIST_STD + MNIST_MEAN
#     patched_unn = patched * MNIST_STD + MNIST_MEAN
#     save_image(x_unn[:num_images], os.path.join(out_dir, f'original_epoch_{epoch}.png'), nrow=8)
#     save_image(patched_unn[:num_images], os.path.join(out_dir, f'patched_epoch_{epoch}.png'), nrow=8)


# def train_patch(args):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

#     # Load pretrained model
#     model = SimpleCNN().to(device)
#     model.load_state_dict(torch.load(args.model_path, map_location=device))
#     model.eval()

#     # Create patch parameter
#     patch_param = nn.Parameter(torch.zeros(1, args.patch_size, args.patch_size, device=device))
#     optimizer = optim.SGD([patch_param], lr=args.lr, momentum=0.9)
#     criterion = nn.CrossEntropyLoss()

#     # Ensure output folder exists
#     os.makedirs(args.out_dir, exist_ok=True)

#     for ep in range(1, args.epochs + 1):
#         running = 0.0
#         for x, y in tqdm(train_loader, desc=f"Patch train ep {ep}/{args.epochs}"):
#             x = x.to(device)
#             B = x.shape[0]
#             p = patch_forward(patch_param)
#             patched, _ = place_patch(x, p, args.patch_size)
#             logits = model(patched)
#             target_labels = torch.full((B,), args.target, dtype=torch.long, device=device)
#             loss = criterion(logits, target_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running += loss.item()

#         # Evaluate fooling rate
#         fool = eval_patch(model, patch_param, test_loader, target_class=args.target,
#                           patch_size=args.patch_size, max_batches=200)
#         print(f"Epoch {ep}: loss {running/len(train_loader):.4f} fool_rate {fool*100:.2f}%")

#         # Visualization every few epochs
#         if ep % args.visualize_every == 0:
#             # Take one batch from test_loader
#             x_vis, _ = next(iter(test_loader))
#             x_vis = x_vis.to(device)
#             p_vis = patch_forward(patch_param)
#             patched_vis, _ = place_patch(x_vis, p_vis, args.patch_size)
#             visualize_patch_batch(x_vis, patched_vis, ep, args.out_dir)

#     # Save patch
#     torch.save(patch_param.detach().cpu(), args.save_path)
#     print('Saved patch to', args.save_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default='mnist_cnn.pth')
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--patch_size', type=int, default=7)
#     parser.add_argument('--lr', type=float, default=0.2)
#     parser.add_argument('--target', type=int, default=2)
#     parser.add_argument('--save_path', type=str, default='mnist_patch.pth')
#     parser.add_argument('--out_dir', type=str, default='./out_patch')
#     parser.add_argument('--visualize_every', type=int, default=5,
#                         help="Save images every N epochs for visualization")
#     args = parser.parse_args()
#     train_patch(args)

# src/train_patch.py
"""
Train adversarial patch with EOT and Lagrangian loss:
    loss = CE_loss (target) + lambda_reg * perceptual_L2_distance

We approximate EOT by sampling `eot_samples` random transforms per batch and averaging losses.
We also save visualization images periodically.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

from src.data import get_dataloaders
from src.model import SimpleCNN, MNIST_MEAN, MNIST_STD
from src.patch import patch_forward, place_patch_and_transform, eval_patch

def unnormalize(tensor):
    return tensor * MNIST_STD + MNIST_MEAN

def visualize_patch_batch(x, patched, epoch, out_dir, num_images=16):
    x_unn = unnormalize(x)
    patched_unn = unnormalize(patched)
    save_image(x_unn[:num_images], os.path.join(out_dir, f'original_epoch_{epoch}.png'), nrow=8)
    save_image(patched_unn[:num_images], os.path.join(out_dir, f'patched_epoch_{epoch}.png'), nrow=8)

def train_patch(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Load pretrained model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Patch param
    patch_param = nn.Parameter(torch.zeros(1, args.patch_size, args.patch_size, device=device))
    optimizer = optim.Adam([patch_param], lr=args.lr) if args.optimizer == 'adam' else optim.SGD([patch_param], lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Patch train ep {ep}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]

            # EOT approximation: sample transforms and average losses
            ce_loss_total = 0.0
            perc_loss_total = 0.0
            for _ in range(args.eot_samples):
                p = patch_forward(patch_param)  # (1,ph,pw)
                patched, _ = place_patch_and_transform(x, p, patch_size=args.patch_size,
                                                       angle_range=(-args.angle_range, args.angle_range),
                                                       translate_frac=args.translate_frac,
                                                       scale_range=(args.scale_low, args.scale_high),
                                                       add_noise_std=args.noise_std)
                logits = model(patched)
                target_labels = torch.full((B,), args.target, dtype=torch.long, device=device)
                ce_loss = criterion(logits, target_labels)
                # perceptual distance: L2 between patched (unnormalized) and original (unnormalized)
                x_unn = unnormalize(x)
                patched_unn = unnormalize(patched)
                perc_loss = torch.mean((patched_unn - x_unn).view(B, -1).pow(2).sum(dim=1))  # mean over batch
                ce_loss_total += ce_loss
                perc_loss_total += perc_loss

            ce_loss_avg = ce_loss_total / float(args.eot_samples)
            perc_loss_avg = perc_loss_total / float(args.eot_samples)

            loss = ce_loss_avg + args.lambda_reg * perc_loss_avg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        # Evaluate
        fool = eval_patch(model, patch_param, test_loader, target_class=args.target,
                          patch_size=args.patch_size, eot_samples=args.eot_samples, device=device)
        print(f"Epoch {ep}: loss {running/len(train_loader):.4f} fool_rate {fool*100:.2f}%")

        # visualize
        if ep % args.visualize_every == 0:
            x_vis, _ = next(iter(test_loader))
            x_vis = x_vis.to(device)
            p_vis = patch_forward(patch_param)
            patched_vis, _ = place_patch_and_transform(x_vis, p_vis, patch_size=args.patch_size,
                                                       angle_range=(-args.angle_range, args.angle_range),
                                                       translate_frac=args.translate_frac,
                                                       scale_range=(args.scale_low, args.scale_high),
                                                       add_noise_std=args.noise_std)
            visualize_patch_batch(x_vis, patched_vis, ep, args.out_dir)

    torch.save(patch_param.detach().cpu(), args.save_path)
    print('Saved patch to', args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='mnist_cnn.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'])
    parser.add_argument('--target', type=int, default=2)
    parser.add_argument('--save_path', type=str, default='mnist_patch.pth')
    parser.add_argument('--out_dir', type=str, default='./out_patch')
    parser.add_argument('--visualize_every', type=int, default=5)
    # EOT / transform params
    parser.add_argument('--eot_samples', type=int, default=4, help='number of random transforms per batch (EOT approx)')
    parser.add_argument('--angle_range', type=int, default=20, help='max absolute rotation degrees')
    parser.add_argument('--translate_frac', type=float, default=0.08, help='max translate fraction of image size')
    parser.add_argument('--scale_low', type=float, default=0.95, help='min scale')
    parser.add_argument('--scale_high', type=float, default=1.05, help='max scale')
    parser.add_argument('--noise_std', type=float, default=0.0, help='add gaussian noise std to transforms')
    # Lagrangian regularizer
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='weight for perceptual (L2) distance term')
    args = parser.parse_args()
    train_patch(args)

