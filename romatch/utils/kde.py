import torch


# def kde(x, std = 0.1, half = True, down = None):
#     # use a gaussian kernel to estimate density
#     if half:
#         x = x.half() # Do it in half precision TODO: remove hardcoding
#     if down is not None:
#         scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
#     else:
#         scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
#     density = scores.sum(dim=-1)
#     return density


def kde(x, std=0.1, half=True, down=1, batch_size=10_000):
    if half:
        x = x.half()
    N = x.shape[0]
    density = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = x[start:end]  # shape [B, D]
        # Compute [B, N] distance matrix between batch and full set
        dists = torch.cdist(x_batch, x[::down])  # shape [B, N]
        scores = torch.exp(-dists**2 / (2 * std**2))  # Gaussian kernel
        density.append(scores.sum(dim=1))  # sum over all reference points

    return torch.cat(density, dim=0)  # [N]