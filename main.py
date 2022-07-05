from PIL import Image
import matplotlib.pyplot as plt

import torch
import hanz
import sys
import pdb
import numpy as np

def get_position_inputs(W, H):
    xs, ys = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W))
    inputs = torch.stack((torch.flatten(xs), torch.flatten(ys)))
    inputs = torch.transpose(inputs, 0, 1)
    max_freq = 20
    N_freqs = 10
    frequencies = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
    frequencies = torch.stack((frequencies, frequencies))
    frequency_x = torch.matmul(inputs, frequencies)
    x = torch.cat((inputs, frequency_x), 1)
    return x

# file_name = sys.argv[1]
file_name = "examples/example-nerf.hanz"
modules = hanz.parseHanz(file_name)

model = modules[0]

image = Image.open("/Users/yzhang/Downloads/puppy2.png")
W, H = image.size

x = get_position_inputs(W, H)
# pdb.set_trace()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

target_image = torch.Tensor(np.asarray(image).flatten()).to(device)

num_training_rounds = 100
x = x.to(device)
for i in range(1,num_training_rounds):
# pdb.set_trace()
    result_image = model(x)
    loss = torch.nn.L1Loss()(result_image.flatten(), target_image )
    print("loss", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def convertResultToImage(result_array, width, height):
    generated_img = result_array.cpu().detach().numpy().astype('uint8')
    generated_img = np.reshape(generated_img, (width, height, 3))
    return generated_img

def showImage(img):
    fig,ax = plt.subplots()
    ax.imshow(img)
    plt.show()


img = convertResultToImage(result_image, W, H)
showImage(img)
