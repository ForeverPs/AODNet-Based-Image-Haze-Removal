import torch
import torch.optim
import numpy as np
from PIL import Image
from model import AODNet
import matplotlib.pyplot as plt


def dehaze_image(image_name):
	data_hazy = Image.open(image_name)
	data_hazy = np.array(data_hazy) / 255.0

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2, 0, 1)
	data_hazy = data_hazy.unsqueeze(0)

	dehaze_net = torch.load('saved_models/dehaze_net_epoch_30.pth', map_location=torch.device('cpu'))

	clean_image = dehaze_net(data_hazy)

	plt.imshow(clean_image.detach().numpy().squeeze())
	plt.show()
	

if __name__ == '__main__':
	img_name = './test_images/canyon.png'
	dehaze_image(img_name)
