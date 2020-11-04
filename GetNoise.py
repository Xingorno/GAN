import torch

def get_noise(n_samples, z_dim, device = 'cpu'):

    return torch.randn(n_samples, z_dim, device = device)


def test_get_noise(n_samples, z_dim, device = 'cpu'):

    noise = get_noise(n_samples, z_dim, device)

    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)


# test input

n_samples = 1000
z_dim = 100

test_get_noise(n_samples, z_dim, 'cpu')

if torch.cuda.is_available():
    test_get_noise(n_samples, 32, 'cuda')

print('Scucess!!!')