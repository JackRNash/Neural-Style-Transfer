import torch
from model import ImageTransNet
from utils import get_img
import matplotlib.pyplot as plt
from torchvision import transforms

'''
Use this class to generate stylized images
MODEL_PATH - directory of the saved model
INPUT_PATH - directory of image to stylize
'''


MODEL_PATH = 'scream.pth.tar'
INPUT_PATH = 'Data/cornell.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_style_transfer(input_path='Data/cornell.jpg', model_path='scream.pth.tar'):
    net = ImageTransNet()
    trained_model = torch.load(model_path, map_location=device)
    net.load_state_dict(trained_model['state_dict'])
    input_img = get_img(input_path)
    prediction = net(input_img)
    plt.imshow(transforms.ToPILImage()(prediction[0].detach()))
    plt.show()


generate_style_transfer(input_path=INPUT_PATH, model_path=MODEL_PATH)
