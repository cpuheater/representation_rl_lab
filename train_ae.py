import argparse
import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from src.autoencoders import Autoencoder, VAE
from src.dataset import DoomImageDataset
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description='train vea or ae')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--grad-clipping', default=None, help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--model-dir', default='trained_models', help='dir saving model')
parser.add_argument('--model-type', default='AE', help='type of model to use: VAE or AE')
parser.add_argument('--latent-dim', default=32, help='latent dim size')
parser.add_argument('--images-dir', default="images", help='')
parser.add_argument('--verbose', default=True, help='verbose')

args = parser.parse_args(args=[])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

width, height = 80, 60
num_channels = 3

exp_name = f'batch_size={args.batch_size}_epoch={args.epochs}_latent_dim={args.latent_dim}_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}_{args.model_type}'


writer = SummaryWriter(f"runs/{exp_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

def save_model(model):
    torch.save(model.state_dict(),
               os.path.join(args.model_dir, f'{exp_name}.pt'))

def create_model():
    if args.model_type == "VAE":
        model = VAE(args.latent_dim).to(device)
    else:
        model = Autoencoder((num_channels, height, width), args.latent_dim).to(device)
    return model

def train_model(epoch, model, optimizer, data_loader, clip_grad_norm=None):
    loss_sum = 0
    num_samples = 0
    for (batch_idx, (img, aug_image)) in enumerate(iter(data_loader)):
        aug_image = aug_image.to(device)
        img = img.to(device)
        optimizer.zero_grad()
        reconstruction = model.forward(aug_image)
        img = img / 255.0
        loss = model.calc_loss(reconstruction, img)
        loss.backward()
        loss_sum += loss.item()
        num_samples += len(img)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
    print(f'Epoch {epoch} Train Loss: {loss_sum}')
    writer.add_scalar("charts/loss", loss_sum, epoch)
    if args.verbose:
        imgs, aug_image = next(iter(data_loader))

        recon = model.get_reconstruction(imgs[0].to(device).unsqueeze(0))
        cv2.imshow("Image", imgs[0].permute(1, 2, 0).cpu().numpy())
        cv2.imshow("Augmentation", aug_image[0].permute(1, 2, 0).cpu().numpy())
        cv2.imshow("Reconstruction", recon[0].cpu().detach().numpy())
        cv2.waitKey(1)
    return

def plot(model, data_loader):
    model.eval()
    imgs = next(iter(data_loader))[0].to(device)
    recon = model.get_reconstruction(imgs)

    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item.permute(1, 2, 0).to('cpu').detach().numpy()[:, :, ::-1])

    for i, item in enumerate(recon[0]):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item.to('cpu').detach().numpy()[:, :, ::-1])
    plt.savefig(os.path.join(args.model_dir, f'{exp_name}.png'))
    plt.show()


def main():
    transform = A.Compose(
        [
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.4),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5),
            ], p=1.0),
            ToTensorV2(),
        ]
    )

    dataset = DoomImageDataset(args.images_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = create_model()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    print(f"DataSet size: {len(dataset)}")
    for epoch in range(args.epochs):
        train_model(epoch, model, optimizer, data_loader, args.grad_clipping)
    plot(model, data_loader)
    save_model(model)

if __name__ == '__main__':
    main()