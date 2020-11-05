import torch
from torch import nn


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace = True),
    )



def test_generator_block(in_features, out_features, num_test = 1000):
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU

    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55 # Post Normalization, SD will be 1, after Relu, SD will be 0.5
    assert test_output.std() < 0.65

test_generator_block(25, 12)
test_generator_block(15, 28)

class Generator(nn.Module):
    def __init__(self, z_dim = 10, im_dim = 784, hidden_dim = 128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),

            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )
    def forward(self, noise):
        return self.gen(noise)
        
    def get_gen(self):
        return self.gen


def test_generator(z_dim, im_dim, hidden_dim, num_test = 10000):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1
    assert test_output.min() > 0
    assert test_output.std() > 0.05
    assert test_output.std() < 0.15

test_generator(5, 10, 20)
test_generator(20, 8, 24)


print("Generator side")

a = torch.randn(2,4)
print(a)