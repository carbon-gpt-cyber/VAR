import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vqvae import VQVAE


class SequenceFolder(Dataset):
    """Dataset loading 1D sequences stored as .npy or .pt tensors."""

    def __init__(self, root: str, split: str = "train"):
        self.files = []
        split_dir = os.path.join(root, split)
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(('.pt', '.npy')):
                self.files.append(os.path.join(split_dir, fname))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        if path.endswith('.npy'):
            seq = torch.from_numpy(np.load(path))
        else:
            seq = torch.load(path)
        if seq.ndim == 1:
            seq = seq.unsqueeze(0)
        return seq.float()


def main():
    parser = argparse.ArgumentParser(description="Train multi-scale VQ-VAE on sequences")
    parser.add_argument('--data_path', type=str, required=True, help='Root folder with train/ and val/ subfolders')
    parser.add_argument('--patch_nums', type=str, default='1,2,3,4,5,6,8,10,13,16',
                        help='Comma separated patch numbers for each scale')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--z_channels', type=int, default=32)
    parser.add_argument('--ch', type=int, default=160)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    patch_nums = tuple(int(x) for x in args.patch_nums.split(','))

    train_set = SequenceFolder(args.data_path, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch,
                  v_patch_nums=patch_nums, in_channels=args.in_channels,
                  is_seq=True, test_mode=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()

    model.train()
    for ep in range(args.epochs):
        for seq in train_loader:
            seq = seq.to(device)
            rec, _, vq_loss = model(seq)
            loss = loss_fn(rec, seq) + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f'Epoch {ep+1}: loss {loss.item():.4f}')

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'vqvae.pth'))


if __name__ == '__main__':
    main()
