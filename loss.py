import torch

'''
The class NSTLoss is used to create a loss function that has the style activations saved(so the style image isn't
repeatedly being run through vgg-16 even though it'll have the same activations every time)
'''

class NSTLoss:

    def __init__(self, style_activations, vgg):
        self.style_activations = style_activations
        self.vgg = vgg

    @staticmethod
    def feature_loss(gen, content):  # averaged loss over all samples in the batch
        n, C, H, W = content.shape
        loss = 0
        for i in range(n):
            loss += 1 / (C * H * W) * (torch.norm(gen[i] - content[i]) ** 2)
        return loss / n

    @staticmethod
    def gram_mat(x):
        C, H, W = x.shape
        x = x.view(C, H * W)
        return 1 / (C * H * W) * torch.mm(x, torch.t(x))

    @staticmethod
    def style_loss(gen, style):
        style_mat = NSTLoss.gram_mat(torch.squeeze(style))  # get rid of batch dimension
        n, C, H, W = gen.shape

        loss = 0
        for i in range(n):
            gen_mat = NSTLoss.gram_mat(gen[i])
            loss += torch.norm(gen_mat - style_mat, p='fro') ** 2
        return loss / n

    def model_loss(self, gen, content, reg_fact=1e-4, content_fact=1, style_fact=[1e3] * 4):
        reg_factor = reg_fact
        content_factor = content_fact
        style_factor = style_fact

        gen_activations = self.vgg(gen)
        content_activations = self.vgg(content)

        loss = content_factor * NSTLoss.feature_loss(gen_activations[1], content_activations[1])  # activations @ relu2_2

        for i, gen_activ in enumerate(gen_activations):
            loss += style_factor[i] * NSTLoss.style_loss(gen_activ, self.style_activations[i])

        # Regularization
        loss += reg_factor * (torch.sum(torch.abs(gen[:, :, :, 1:] - gen[:, :, :, :-1]))
                              + torch.sum(torch.abs(gen[:, :, 1:, :] - gen[:, :, :-1, :])))

        loss *= 1e4
        # print('loss:', loss.item())
        return loss
