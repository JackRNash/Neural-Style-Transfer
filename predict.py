import torch
from model import ImageTransNet
from utils import get_img
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageSequence

'''
Use this class to generate stylized images
MODEL_PATH - directory of the saved model (ex. 'starry_night.pth.tar' & 'scream.pth.tar')
INPUT_PATH - directory of image or gif to stylize (ex. 'Data/cornell.jpg' or 'Data/sky.gif')
'''


MODEL_PATH = 'scream.pth.tar'
INPUT_PATH = 'Data/cornell.jpg'
VERBOSE = False  # Print an indicator that each frame has been stylized (for gifs only)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_style_transfer_img(input_path='Data/cornell.jpg', model_path='scream.pth.tar'):
    input_img = get_img(input_path, resize=False)
    net = ImageTransNet(input_img.shape[2:])
    trained_model = torch.load(model_path, map_location=device)
    net.load_state_dict(trained_model['state_dict'])
    prediction = net(input_img)
    img = transforms.ToPILImage()(prediction[0].detach())

    name = input_path[:input_path.find('.')]
    save_name = name + '_stylized_' + model_path[:model_path.find('.pth')] + '.png'
    img.save(save_name)

    plt.imshow(img)
    plt.show()


def generate_style_transfer_gif(input_path, model_path='scream.pth.tar', verbose=0):
    gif = Image.open(input_path)

    gif_tensors = []
    preprocess = transforms.Compose([transforms.ToTensor()])
    net = ImageTransNet(preprocess(gif).shape[1:])
    trained_model = torch.load(model_path, map_location=device)
    net.load_state_dict(trained_model['state_dict'])
    # print('loaded network')
    i = 1
    for frame in ImageSequence.Iterator(gif):
        if verbose:
            print('Stylizing frame', i)
        with torch.no_grad(): # Don't track gradients for back prop since this is inference
            img = preprocess(frame.convert('RGB')) # Load as RGB image
            img = torch.unsqueeze(img, 0) # Add fake batch dimension
            gif_tensors.append(torch.squeeze(net(img))) # Get stylized frame, add after removing batch dimension
        i += 1

    # Prep to save the gif
    frames = [transforms.ToPILImage()(torch.squeeze(tensor)) for tensor in gif_tensors]
    output_gif = frames[0]
    name = input_path[:-4]
    save_name = name + '_stylized_' + model_path[:model_path.find('.pth')] + '.gif'

    # Save the gif
    output_gif.save(save_name, save_all=True, append_images=frames[0:], disposal=2, loop=0)
    print('Gif saved at', save_name)


if INPUT_PATH[-3:] == 'gif':
    generate_style_transfer_gif(INPUT_PATH, model_path=MODEL_PATH, verbose=VERBOSE)
else:
    generate_style_transfer_img(input_path=INPUT_PATH, model_path=MODEL_PATH)

