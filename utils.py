import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

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
            discriminator.zero_grad()
            lossD.backward(retain_graph=True)
            optim_disc.step()

            # Train Generator
            output = discriminator(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            optim_gen.step()

            if batch_idx == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

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

    return lossD_list, lossG_list

def train_dcgan():
    pass