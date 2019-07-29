from utils import get_img, get_truncated_vgg
from loss import NSTLoss
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

'''
Gatys et al implementation of neural style transfer (VERY slow on CPUs)
'''

class slowNST():

    def __init__(self, content_path='Data/cornell.jpg', style_path='Data/starry-night.jpg'):
        self.content = get_img(content_path)
        self.style = get_img(style_path)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.content = self.content.cuda()
            self.style = self.style.cuda()
        vgg = get_truncated_vgg()
        self.style_activations = vgg(self.style)
        self.loss_fn = NSTLoss(self.style_activations, vgg)

    def generate_image(self):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        prediction = torch.randn(1, 3, 256, 256).clone().detach().requires_grad_(True)
        if self.use_cuda:
            prediction = prediction.cuda()
        optim = torch.optim.LBFGS([prediction], lr=1)

        iteration = 0

        def closure():
            # global iteration
            optim.zero_grad()
            loss = self.loss_fn.model_loss(prediction, self.content, reg_fact=1e-5, content_fact=1e1,
                              style_fact=[1e3] * 4)  # change to e3/e4, maybe reduce lr
            loss.backward(retain_graph=True)

            # if iteration % 10 == 0:
            #     print('Iteration:', iteration, 'Loss', loss.item())
            #     if self.use_cuda:
            #         plt.imshow(transforms.ToPILImage()(prediction[0].cpu().detach()))
            #     else:
            #         plt.imshow(transforms.ToPILImage()(prediction[0]))
            #     plt.show()
            # iteration += 1
            return loss

        for epoch in range(100):  # 20 iterations per round, 25 rounds = 500 iterations total
            print('Round:', epoch*20)
            # if self.use_cuda:
            #     plt.imshow(transforms.ToPILImage()(prediction[0].cpu().detach()))
            # else:
            #     plt.imshow(transforms.ToPILImage()(prediction[0]))
            # plt.show()
            optim.step(closure)

        prediction = torch.clamp(prediction, 0, 255)
        if self.use_cuda:
            plt.imshow(transforms.ToPILImage()(prediction[0].cpu().detach()))
        else:
            plt.imshow(transforms.ToPILImage()(prediction[0]))


if __name__ == '__main__':
    # Set the style and content images
    content = 'Data/cornell.jpg'
    style = 'Data/starry-night.jpg'
    model = slowNST(content, style)
    model.generate_image()

