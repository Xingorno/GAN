import torch
import PyQt5
from torch import nn
from tqdm.auto import tqdm # show progress
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
torch.manual_seed(0) # Set for testing purposes
print(matplotlib.get_backend())
from Discriminator import Discriminator
from Generator import Generator
from GetNoise import get_noise

def show_tensor_images(image_tensor, num_images = 25, size = (1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow = 5)
    print(image_grid.shape)
    print(matplotlib.get_backend())
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



#print("hello world")

# Set your parameters

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001
device = 'cpu'
dataloader = DataLoader(MNIST('.', download = False, transform = transforms.ToTensor()),\
    batch_size = 128, shuffle = True) 

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)

disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr= lr)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    image = gen(noise).detach()
    output_fake = disc(image)
    ground_truth_fake = torch.zeros_like(output_fake)
    loss_fake_images = criterion(output_fake, ground_truth_fake)

    output_real = disc(real)
    ground_truth_real = torch.ones_like(output_real)
    loss_real_images = criterion(output_real, ground_truth_real)

    disc_loss = (loss_fake_images + loss_real_images) /2
    return disc_loss

def test_disc_reasonable(num_images = 10):
    import inspect, re
    lines = inspect.getsource(get_disc_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None

    z_dim = 64
    gen = torch.zeros_like
    disc = lambda x: x.mean(1)[:, None]
    criterion = torch.mul
    real = torch.ones(num_images, z_dim)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(disc_loss.mean() - 0.5) < 1e-5)

    gen = torch.ones_like
    criterion = torch.mul
    real = torch.zeros(num_images, z_dim)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')) < 1e-5)

    gen = lambda x: torch.ones(num_images, 10)
    disc = lambda x: x.mean(1)[:, None] + 10
    criterion = torch.mul
    real = torch.zeros(num_images, 10)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean() -5 ) < 1e-5)

    gen = torch.ones_like
    disc = nn.Linear(64, 1, bias = False)
    real = torch.ones(num_images, 64) * 0.5
    disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
    disc_opt = torch.optim.Adam(disc.parameters(), lr = lr)
    criterion = lambda x, y: torch.sum(x) + torch.sum(y)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean()
    disc_loss.backward()
    assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))

def test_disc_loss(max_tests = 10):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr= lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr= lr)
    num_steps = 0

    for real, _ in dataloader:
        cur_bath_size = len(real)
        real = real.view(cur_bath_size, -1).to(device)
        # Zero out the gradient before backpropagation
        disc_opt.zero_grad()

        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_bath_size, z_dim, device)
        assert (disc_loss - 0.68).abs() < 0.05
        # Update the gradients
        disc_loss.backward(retain_graph = True)

        assert gen.gen[0][0].weight.grad is None

        old_weight = disc.disc[0][0].weight.data.clone()
        # Updata optimizer
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data

        assert not torch.all(torch.eq(old_weight, new_weight))
        num_steps += 1
        if num_steps >= max_tests:
            break

test_disc_reasonable()
test_disc_loss()

    

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    image = gen(noise)
    output_fake = disc(image)
    ground_truth_fake = torch.zeros_like(output_fake)
    gen_loss = criterion(output_fake, ground_truth_fake)
    return gen_loss

def test_gen_reasonable(num_images = 10):
    import inspect, re
    lines = inspect.getsource(get_gen_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    assert tuple(gen_loss_tensor.shape) ==(num_images, z_dim)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul
    real = torch.zeros(num_images, 1)
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    #assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)

def test_gen_loss(num_images):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr = lr)
    gen_loss = get_gen_loss(gen, disc, criterion, num_images, z_dim, device)
    assert (gen_loss - 0.7).abs() < 0.1
    # Update the gradient
    gen_loss.backward()
    old_weight = gen.gen[0][0].weight.clone()
    # Update optimizer
    gen_opt.step()
    new_weight = gen.gen[0][0].weight
    assert not torch.all(torch.eq(old_weight, new_weight))

test_gen_reasonable(10)
test_gen_loss(18)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False
for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)
        # Zero out the gradients of the disciminator
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        # Update the gradients
        disc_loss.backward(retain_graph = True)
        # Update optimizer
        disc_opt.step()
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()
        # Zero out the gradients of generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        # Update the grdients of generator
        gen_loss.backward(retain_graph = True)
        # Update the optimizer
        gen_opt.step()
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed!")
        
        mean_discriminator_loss += disc_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step  > 0:
            print(f"Epoch {epoch}, step{cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device= device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1














    