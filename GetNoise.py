import torch
#import PyQt5
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt 
def get_noise(n_samples, z_dim, device = 'cpu'):

    return torch.randn(n_samples, z_dim, device = device)


def test_get_noise(n_samples, z_dim, device = 'cpu'):

    noise = get_noise(n_samples, z_dim, device)
    plt.figure()
    plt.imshow(noise)
    plt.show()
    # output shape/size
    assert tuple(noise.shape) == (n_samples, z_dim)
    # output value (roughly range)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    # outout device detecting
    assert str(noise.device).startswith(device)
    #print(str(noise.device))

# test input

n_samples = 1000
z_dim = 100

test_get_noise(n_samples, z_dim, 'cpu')

if torch.cuda.is_available():
    test_get_noise(n_samples, 32, 'cuda')

print('Scuccess!!!')