import re

import torch
import torch.nn.functional as F


def save_model(filename, model, optimizer, iter, args, logdir):
    data = {'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'iter': iter,
            'args': args,
            'logdir': logdir,
            }
    torch.save(data, filename)


def load_model(filename):
    data = torch.load(filename)
    return data['model'], data['optim'], data['iter'], data['args'], data['logdir']

