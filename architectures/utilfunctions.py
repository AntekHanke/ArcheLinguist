import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_pairwise_differences(tensor):
    logger.debug("Got tesnor: ", tensor.size())
    batch_size,x,y,z = tensor.size()
    pairs = [(i, j) for i in range(batch_size) for j in range(batch_size) if i < j]

    pairwise_diffs = torch.empty((len(pairs),x,y,z))
    for idx, (i, j) in enumerate(pairs):
        pairwise_diffs[idx] = tensor[i] - tensor[j]
    return pairwise_diffs