import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

def train_fcgan(generator, discriminator, loader, device, num_epochs=100, z_dimension=64, lr=3e-4):
    optim_disc = optim.Adam(discriminator.parameters(), lr=lr)
    optim_gen = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    writer_fake = SummaryWriter(log_dir=f'./runs/GAN_MNIST/fake')
    writer_real = SummaryWriter(log_dir=f'./runs/GAN_MNIST/real')
    
    step = 0
    lossG_list, lossD_list = [], []

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(batch_size, z_dimension).to(device)
            fake = generator(noise)
            disc_real = discriminator(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            optim_disc.zero_grad()
            lossD.backward(retain_graph=True) #retain_graph=True
            optim_disc.step()

            # Train Generator
            output = discriminator(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            optim_gen.zero_grad()
            lossG.backward()
            optim_gen.step()

            if batch_idx  == len(loader)-1:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)-1} Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

                with torch.no_grad():
                    fake = generator(torch.randn((batch_size, z_dimension)).to(device)).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                    writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)
                    step += 1

                lossD_list.append(lossD.item())
                lossG_list.append(lossG.item())

    plot_loss_curves(lossD_list, lossG_list)
    writer_fake.close()
    writer_real.close()

def train_dcgan(generator, discriminator, loader, device, num_epochs=5, z_dim=100, lr_disc=1e-4, lr_gen=2e-4):
    
    optim_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
    optim_gen = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    criterion = nn.BCELoss()


    writer_fake = SummaryWriter(log_dir='./runs/DCGAN_CIFAR10/fake')
    writer_real = SummaryWriter(log_dir='./runs/DCGAN_CIFAR10/real')

    step = 0
    lossG_list, lossD_list = [], []

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            batch_size = real.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = generator(noise)
            disc_real = discriminator(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.full_like(disc_real, 0.9))  # Label smoothing
            disc_fake = discriminator(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            optim_disc.zero_grad()
            lossD.backward()
            optim_disc.step()

            # Train Generator
            output = discriminator(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            optim_gen.zero_grad()
            lossG.backward()
            optim_gen.step()

    
            if batch_idx == len(loader)-1:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)-1} "
                    f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = generator(torch.randn((batch_size, z_dim, 1, 1)).to(device))
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                    writer_fake.add_image("DCGAN Fake Images", img_grid_fake, global_step=step)
                    writer_real.add_image("DCGAN Real Images", img_grid_real, global_step=step)
                
                step += 1

                lossD_list.append(lossD.item())
                lossG_list.append(lossG.item())

    plot_loss_curves(lossD_list, lossG_list)
    writer_fake.close()
    writer_real.close()

def plot_loss_curves(lossD_list, lossG_list):
    plt.figure(figsize=(10, 5))
    plt.plot(lossD_list, label="Discriminator Loss")
    plt.plot(lossG_list, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves for Generator and Discriminator")
    plt.legend()
    plt.show()

def latent_space_interpolation(generator, z_dim, num_steps=10, device='cuda'):
    # Ensure the generator is on the same device as the input
    generator.to(device)
    
    # Generate two distinct random latent vectors
    z1 = torch.randn(1, z_dim).to(device)
    z2 = torch.randn(1, z_dim).to(device)
    
    # Linear interpolation between z1 and z2
    interpolation_steps = torch.linspace(0, 1, num_steps).to(device)
    interpolated_latents = [(1 - alpha) * z1 + alpha * z2 for alpha in interpolation_steps]
    
    # Generate images for each interpolated latent vector
    generated_images = [generator(latent).view(28, 28).cpu().detach().numpy() for latent in interpolated_latents]
    
    # Plotting the interpolation results with improved layout
    fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle("Latent Space Interpolation (Refined)")
    plt.show()

    
def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
    
    # KL-divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # ELBO loss
    return recon_loss + kl_div, recon_loss, kl_div
