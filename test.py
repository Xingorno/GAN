
import matplotlib as mpl
# mpl.use('Qt5Agg')
# mpl.rcParams['backend'] = 'Qt5Agg'
# print(mpl.get_backend())
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
# mpl.rcParams['backend'] = 'Qt5Agg'
def get_noise(n_samples, z_dim, device = 'cpu'):

    return torch.randn(n_samples, z_dim, device = device)


n_samples = 600
z_dim = 400

image = get_noise(n_samples, z_dim, 'cpu')

# print(image)

plt.figure()
plt.imshow(image)
plt.show() 
print(mpl.matplotlib_fname())
print(mpl.get_backend())
# print(mpl.is_interactive())

#plt.savefig("savedImage.png")
img = mpimg.imread('savedImage.png')
# print(img)
plt.imshow(img)
plt.show()