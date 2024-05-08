import torch.nn as nn
from torchvision.transforms import RandomPerspective
from torchvision.transforms import GaussianBlur
import torch
from typing import List, Callable


# class Perturbation:
#     def __call__(self, image: torch.Tensor) -> torch.Tensor:
#         pass


default_perturbs_0 = [RandomPerspective(distortion_scale = 0.6, p=1.0)]
default_perturbs_1 = [RandomPerspective(distortion_scale = 0.6, p=1.0), GaussianBlur(kernel_size=5, sigma=1.5)]

# if (0 == 1):
#     image = image + (
#                 torch.rand(image.size()).to(torch.device("cuda")) - torch.tensor(0.5).to(torch.device("cuda"))) / 10
# if (0 == 1):
#     # print("perturber image size: ", image.size())
#     image = image + (torch.rand(image.size()[:1]).to(torch.device("cuda")) - torch.tensor(0.5).to(
#         torch.device("cuda")))[:, None, None, None] / 100


class Perturber(nn.Module):
    def __init__(self, perturbs: List[Callable[[torch.Tensor], torch.Tensor]] = default_perturbs_0):
        super(Perturber, self).__init__()
        self.perturbs = perturbs

    def forward(self, image):

        for p in self.perturbs:
            image = p(image)
        return image