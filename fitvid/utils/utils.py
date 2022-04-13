import torch


def resize_tensor(t, dims):
    h, w = dims
    if t.shape[-2] == h and t.shape[-1] == w:
        return t
    else:
        # uses Bilinear interpolation by default, use antialiasing
        orig_shape = tuple(t.shape[:-3])
        img_shape = tuple(t.shape[-3:])
        t = torch.reshape(t, (-1,) + img_shape)
        t = Resize(dims, antialias=True)(t)
        t = torch.reshape(t, orig_shape + (img_shape[0],) + tuple(dims))
        return t


def dict_to_cuda(d):
    # turn all pytorch tensors into cuda tensors, leave all other objects along
    d = {k: v.cuda() if torch.is_tensor(v) else v for k, v in d.items()}
    return d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)