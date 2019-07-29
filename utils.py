from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision

'''
A collection of useful methods
'''

def get_img(path='Data/starry-night.jpg'):
    img = Image.open(path)
    img = img.resize((256, 256))
    plt.imshow(img)
    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    return img


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# Used to initialize the weights for a model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


'''
Load vgg-16
Code from:
https://discuss.pytorch.org/t/accessing-intermediate-layers-of-a-pretrained-network-forward/12113/2
Rewrite someday(rewrite solely for originality, code works perfectly well so no rush)
'''
class vggTruncated(nn.Module):

    def __init__(self):
        super(vggTruncated, self).__init__()
        features = list(torchvision.models.vgg16(pretrained=True).features)[
                   :23]  # Last layer is layer 22, need nothing else
        self.layers = nn.ModuleList(features).eval()

    def forward(self, x):
        activations = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in {3, 8, 15, 22}:  # relu1_2, relu2_2, relu3_3, and relu 4_3 respectively
                activations.append(x)  # actvation at this layer
        return activations


# Returns a truncated vgg that outputs the specific activations needed
def get_truncated_vgg():
    vgg = vggTruncated()
    if torch.cuda.is_available():
        vgg = vgg.cuda()
    for param in vgg.parameters():  # Freeze
        param.requires_grad_(False)
    return vgg
