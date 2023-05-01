import torch

def white_noise(z, snr_db):
    z_clone = torch.detach(z)
    x_power = torch.Tensor([1.]) / torch.numel(z_clone)
    snr_lin = 10 ** (snr_db / 10)
    var = x_power / snr_lin
    std = torch.sqrt(var)
    return torch.randn_like(z_clone) * std

x = torch.zeros(100)
x[1]=1

print(white_noise(x, 40).abs().mean())