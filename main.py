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

# the scene is at (0,0,0) location. The camera is around the scene.
# camera_pos is there the viewer is;
# screen_center is where the camera is pointing at;
# The sample view direction is parallel to the ground.
#  (meaning: camera_pos.y = screen_center.y)
def getViewDirs(camera_pos, screen_center, width, height):
    center_vect = screen_center - camera_pos
    y_dir = np.array([0,1,0])
    x_dir = np.cross(camera_pos, y_dir)
    x_dir = x_dir/ np.linalg.norm(x_dir)
    view_dirs = []
    for y in range(-height//2, height - height//2):
        for x in range(-width//2, width - width//2):
            dir = center_vect + x * x_dir + y * y_dir
            dir = dir/ np.linalg.norm(dir)
            view_dirs.append(dir)
    return view_dirs

# file_name = sys.argv[1]
file_name = "./examples/example-nerf-with-view-dir.hanz"
modules = hanz.parseHanz(file_name)

model = modules[0]

image = Image.open("/Users/yzhang/Downloads/puppy2.png")
image1 = Image.open("/Users/yzhang/Downloads/puppy1.png")
W, H = image.size

x = get_position_inputs(W, H)
# pdb.set_trace()

camera_pos_0 = np.array([0, 0, -360])
screen_center_0 = np.array([0, 0, -180])
camera_pos_1 = np.array([255, 0, -255])
screen_center_1 = np.array([127.5, 0, -127.5])

ray_origins_0 = torch.Tensor([camera_pos_0] * (W * H))
ray_origins_1 = torch.Tensor([camera_pos_1] * (W * H))
ray_dirs_0 = torch.Tensor(getViewDirs(camera_pos_0, screen_center_0, W, H))
ray_dirs_1 = torch.Tensor(getViewDirs(camera_pos_1, screen_center_1, W, H))

input0 = torch.cat((x, ray_origins_0, ray_dirs_0), 1)
input1 = torch.cat((x, ray_origins_1, ray_dirs_1), 1)
inputs = torch.cat((input0, input1), 0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

target_image = torch.Tensor(np.asarray(image).flatten())
target_image1 = torch.Tensor(np.asarray(image1).flatten())
target_image = torch.cat((target_image, target_image1), 0).to(device)

num_training_rounds = 2
inputs = inputs.to(device)
for i in range(1,num_training_rounds):
# pdb.set_trace()
    result_image = model(inputs)
    loss = torch.nn.L1Loss()(result_image.flatten(), target_image )
    print("{}/{} loss".format(i, num_training_rounds), loss)
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


img = convertResultToImage(result_image[0:W*H], W, H)
img2 = convertResultToImage(result_image[W*H:], W, H)
showImage(img)
showImage(img2)
