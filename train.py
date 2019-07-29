import torch
import torchvision
from torchvision import transforms
from model import ImageTransNet
from utils import init_weights, save_checkpoint, get_img, get_truncated_vgg
from loss import NSTLoss
import matplotlib.pyplot as plt

'''
Run this class to train the a network on a specified style img. NOTE: the mscoco dataset is not included and if you
wish to train this model, you must download it and put the images in Data/mscoco/train2014/train2014. There is an
example image in it currently to allow the script to run. It should be deleted if you intend on actually 
training this model

Change STYLE_IMG to the path of the desired image. Some example choices are available in Data in starry-night.jpg, 
scream.jpg, and waves.jpg
Change CHECKPOINT to the path of a saved checkpoint of the model to resume training. Set it as None to have it start
from the beginning
'''

STYLE_IMG = 'Data/starry-night.jpg'  # E.g. Data/starry-night.jpg
CHECKPOINT = None  # E.g. 'checkpoint.pth.tar'

use_cuda = torch.cuda.is_available()


'''
Train the model. If training from a previous checkpoint, specify the path to said checkpoint and the training
will pick up where it left off
'''
def train(checkpoint_path=None, style_path='Data/starry-night.jpg'):
    torch.manual_seed(0)  # for reproducibility

    # LOAD THE DATA !
    preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    mscoco = torchvision.datasets.ImageFolder('Data/mscoco/train2014', transform=preprocess)
    data_loader = torch.utils.data.DataLoader(mscoco, batch_size=4)

    # Build the network
    net = ImageTransNet()
    if use_cuda:
        net = net.cuda()

    # Initialize the weights
    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    start_epoch = 0

    # Load from previous point
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Get the style image & its VGG activations
    style = get_img(style_path)
    vgg = get_truncated_vgg()
    style_activations = vgg(style)

    # Get the loss function
    nst = NSTLoss(style_activations, vgg)

    # Train model
    for epoch in range(start_epoch, 2):

        for batch_index, data in enumerate(data_loader):
            data = data[0]  # data[1] is the 'class', which is always 0, just want the tensors
            if use_cuda:
                data = data.cuda()
            inputs = data
            content = data

            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = nst.model_loss(outputs, content, reg_fact=1e-3, content_fact=5e1, style_fact=[1e3] * 4)
            loss.backward()
            optimizer.step()

            # Occasionally show image # Works well in a jupyter notebook, not so much in a .py file b/c .show()
            # stops execution
            # if batch_index % 10 == 0:
            #     print('Image:', 1 + 4 * batch_index, 'Loss:', loss.item())
            #     if use_cuda:
            #         plt.imshow(transforms.ToPILImage()(
            #             torchvision.utils.make_grid([data[0].cpu().detach(), outputs[0].cpu().detach()], nrow=2)))
            #     else:
            #         plt.imshow(transforms.ToPILImage()(
            #             torchvision.utils.make_grid([data[0].detach(), outputs[0].detach()], nrow=2)))
            #     plt.show()

            del loss  # Might not help, but can't hurt to free up some memory(had some memory leak issues early on)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        print('Epoch:', epoch + 1, '\n\nSAVED A NEW CHECKPOINT\n\n')


train(checkpoint_path=CHECKPOINT, style_path=STYLE_IMG)
