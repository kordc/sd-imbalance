import os
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from torch.utils.tensorboard import SummaryWriter  # Removed TensorBoard
import torchvision.utils as vutils
import wandb  # Added wandb


# --- Configuration ---
real_data_dir = "cifar10_raw/cat"
synthetic_data_dir = "flux_tests_32_bicubic"
checkpoint_dir = "checkpoints"
generated_images_dir = "generated_images"
image_size = 32  # CIFAR-10 image size
batch_size = 16  # Adjust based on your GPU memory
num_epochs = 200
lr = 0.0002
beta1 = 0.5
lambda_cycle = 10.0
lambda_identity = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(generated_images_dir, exist_ok=True)

# --- wandb Initialization ---
wandb.init(
    project="cyclegan-cats",
    config={  # Changed project name
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lambda_cycle": lambda_cycle,
        "lambda_identity": lambda_identity,
        "image_size": image_size,
        "dataset": "CIFAR-10 Cats vs Synthetic",
        "architecture": "CycleGAN",
    },
)
config = wandb.config

# --- Data Loading ---
# Data augmentation and normalization for training
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Check if directories exist
if not os.path.exists(real_data_dir) or not os.path.exists(synthetic_data_dir):
    raise FileNotFoundError("Data directories not found.  Check paths.")

# Create datasets
real_dataset = datasets.ImageFolder(
    root=os.path.dirname(real_data_dir), transform=transform
)
synthetic_dataset = datasets.ImageFolder(root=synthetic_data_dir, transform=transform)


# Create data loaders
real_loader = DataLoader(
    real_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
)
synthetic_loader = DataLoader(
    synthetic_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

# --- Model Definitions ---


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),  # No stride here
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),  # Output 1 channel
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# --- Image Buffer ---
class ImageBuffer:
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.num_imgs = 0
            self.buffer = []

    def push_and_pop(self, images):
        if self.buffer_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.num_imgs = self.num_imgs + 1
                self.buffer.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.buffer_size - 1)
                    tmp = self.buffer[random_id].clone()
                    self.buffer[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


# --- Initialization ---


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Instantiate models
netG_A2B = Generator().to(device)  # Synthetic to Real
netG_B2A = Generator().to(device)  # Real to Synthetic
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

# --- Loss Functions ---
criterion_GAN = nn.MSELoss()  # Adversarial loss
criterion_cycle = nn.L1Loss()  # Cycle consistency loss
criterion_identity = nn.L1Loss()  # Identity loss

# --- Optimizers ---
optimizer_G = optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
    lr=lr,
    betas=(beta1, 0.999),
)
optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(beta1, 0.999))


# --- Learning Rate Scheduler ---
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch - 100) / float(100 + 1)
    return lr_l


scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)


# --- Image Buffer ---
fake_A_buffer = ImageBuffer()
fake_B_buffer = ImageBuffer()

# --- Removed TensorBoard Setup ---
# writer = SummaryWriter()

# --- Training Loop ---

for epoch in range(num_epochs):
    for i, (real_A, real_B) in enumerate(zip(synthetic_loader, real_loader)):
        real_A = real_A[0].to(device)  # Image
        real_B = real_B[0].to(device)

        # --- Train Generators ---
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * lambda_identity

        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * lambda_identity

        # GAN loss (A to B)
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

        # GAN loss (B to A)
        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * lambda_cycle

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * lambda_cycle

        # Total generator loss
        loss_G = (
            loss_GAN_A2B
            + loss_GAN_B2A
            + loss_cycle_ABA
            + loss_cycle_BAB
            + loss_identity_A
            + loss_identity_B
        )
        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminator A ---
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

        # Total discriminator loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # --- Train Discriminator B ---
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))

        # Total discriminator loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(real_loader)}], Loss_G: {loss_G.item():.4f}, Loss_D_A: {loss_D_A.item():.4f}, Loss_D_B: {loss_D_B.item():.4f}"
            )

            # --- Log to wandb ---
            wandb.log(
                {
                    "Loss/G": loss_G.item(),
                    "Loss/D_A": loss_D_A.item(),
                    "Loss/D_B": loss_D_B.item(),
                    "Loss/G_GAN": (loss_GAN_A2B + loss_GAN_B2A).item(),
                    "Loss/G_Cycle": (loss_cycle_ABA + loss_cycle_BAB).item(),
                    "Loss/G_Identity": (loss_identity_A + loss_identity_B).item(),
                    "epoch": epoch,  # Log the epoch
                    "batch": i,  # Log the batch number
                }
            )

            # Log generated images to wandb
            with torch.no_grad():
                fake_B_sample = netG_A2B(real_A)
                fake_A_sample = netG_B2A(real_B)

            img_grid_real_A = vutils.make_grid(real_A, normalize=True)
            img_grid_fake_B = vutils.make_grid(fake_B_sample, normalize=True)
            img_grid_real_B = vutils.make_grid(real_B, normalize=True)
            img_grid_fake_A = vutils.make_grid(fake_A_sample, normalize=True)

            # Convert grids to images for wandb
            wandb.log(
                {
                    "Real/A": [wandb.Image(img_grid_real_A)],
                    "Fake/B": [wandb.Image(img_grid_fake_B)],
                    "Real/B": [wandb.Image(img_grid_real_B)],
                    "Fake/A": [wandb.Image(img_grid_fake_A)],
                }
            )

    # Update learning rates
    scheduler_G.step()
    scheduler_D_A.step()
    scheduler_D_B.step()

    # Save model checkpoints
    if (epoch + 1) % 25 == 0:
        torch.save(
            netG_A2B.state_dict(),
            os.path.join(checkpoint_dir, f"netG_A2B_epoch_{epoch + 1}.pth"),
        )
        torch.save(
            netG_B2A.state_dict(),
            os.path.join(checkpoint_dir, f"netG_B2A_epoch_{epoch + 1}.pth"),
        )
        torch.save(
            netD_A.state_dict(),
            os.path.join(checkpoint_dir, f"netD_A_epoch_{epoch + 1}.pth"),
        )
        torch.save(
            netD_B.state_dict(),
            os.path.join(checkpoint_dir, f"netD_B_epoch_{epoch + 1}.pth"),
        )

# writer.close()  # Removed TensorBoard close
wandb.finish()  # Finish wandb run


# --- Inference ---
# Load the trained model
netG_A2B.load_state_dict(
    torch.load(os.path.join(checkpoint_dir, f"netG_A2B_epoch_{num_epochs}.pth"))
)
netG_A2B.eval()  # Set the model to evaluation mode

# Prepare a data loader for the synthetic images (no need to shuffle)
inference_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

inference_dataset = datasets.ImageFolder(
    root=synthetic_data_dir, transform=inference_transform
)
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)


# Generate and save images
with torch.no_grad():  # Disable gradient calculation during inference
    for i, (synthetic_img, _) in enumerate(inference_loader):
        synthetic_img = synthetic_img.to(device)
        fake_real_img = netG_A2B(synthetic_img)

        # Denormalize the image (inverse of the normalization)
        fake_real_img = (fake_real_img * 0.5) + 0.5
        fake_real_img = fake_real_img.squeeze(0)  # Remove batch dimension

        # Convert to PIL image and save
        fake_real_img = transforms.ToPILImage()(fake_real_img.cpu())
        fake_real_img.save(os.path.join(generated_images_dir, f"generated_{i}.png"))

print(f"Generated images saved to: {generated_images_dir}")
