import torch
from torch import nn


def get_discriminator_block(input_dim, output_dim):
    
    return nn.Sequential(

        nn.Linear(input_dim, output_dim), # after linear transform, the std will be 0.6 around

        nn.LeakyReLU(negative_slope = 0.2, inplace= True),
    )

def test_disc_block(input_features, output_features, num_test = 10000):

    disc = get_discriminator_block(input_features, output_features)

    assert len(disc) == 2
    test_input = torch.randn(num_test, input_features)
    test_output = disc(test_input)

    assert tuple(test_output.shape) == (num_test, output_features)
    
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5
    #print(test_output.min())
    #print(test_output.max())
    #print(test_output.std())

test_disc_block(25, 12)
test_disc_block(15, 28)


class Discriminator(nn.Module):
    def __init__(self, im_dim = 784, hidden_dim = 128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.disc

def test_discriminator(im_dim, hidden_dim, num_test = 100):
    disc = Discriminator(im_dim, hidden_dim).get_disc()
    assert len(disc) == 4

    test_input = torch.randn(num_test, im_dim)
    test_output = disc(test_input)

    assert tuple(test_output.shape) == (num_test, 1)
    assert test_input.max() > 1
    assert test_input.min() < -1

test_discriminator(5, 10)
test_discriminator(20, 8)
print("Success!!!")
