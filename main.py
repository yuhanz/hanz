from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import hanz_util
import sys
import pdb
import numpy as np
import math
import time

def get_position_inputs(W, H, max_freq = 10, N_freqs = 10):
    frequencies = np.array(list(map(lambda x: 1- 0.65**x, np.arange(0,N_freqs)))) * 2.**max_freq
    # frequencies = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
    # frequencies[0] = 1.0
    # frequencies[-1] = 2.**max_freq
    # frequencies = torch.Tensor(frequencies)
    # frequencies = torch.stack((frequencies, frequencies))
    #xs, ys = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W))
    xs, ys = torch.meshgrid(torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W))
    frequencies_x = torch.Tensor(np.array(list(map(lambda f: f * xs.numpy(), frequencies))), requires_grad=False)
    frequencies_y = torch.Tensor(np.array(list(map(lambda f: f * ys.numpy(), frequencies))), requires_grad=False)
    inputs = torch.stack((torch.flatten(xs), torch.flatten(ys)))
    inputs = torch.transpose(inputs, 0, 1)
    fx = torch.transpose(frequencies_x.flatten(1), 0, 1)
    fy = torch.transpose(frequencies_y.flatten(1), 0, 1)
    x = torch.cat((inputs, fx, fy), 1)
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

def rotateCameraAroundOrigin(angle, screen_distance = 180):
  camera_distance = -360
  screen_distance = -screen_distance
  cosA = np.cos(angle)
  sinA = np.sin(angle)
  camera_pos = np.array([sinA, 0, cosA]) * camera_distance
  screen_center = np.array([sinA, 0, cosA]) * screen_distance
  return [camera_pos, screen_center]

def convertRayDirToAngles(ray_dir):
    d = np.linalg.norm(ray_dir[0:2])
    a1 = math.acos(ray_dir[0] / d) * (0 if ray_dir[1] == 0 else ray_dir[1] / abs(ray_dir[1]))
    a2 = math.acos(ray_dir[2])
    return [a1,a2]

def getInputFromCameraPosition(angle, screen_width, screen_height, view_port_range, num_voxel_samples = 7, screen_distance = 180):
  W = screen_width
  H = screen_height
  camera_pos, screen_center = rotateCameraAroundOrigin(angle, screen_distance = 180)
  ray_dirs = torch.Tensor(getViewDirs(camera_pos, screen_center, W, H), requires_grad = False)
  voxel_points = []
  i = 0
  for ray_dir in ray_dirs:
      i = i +1
      print('---{}/{}'.format(i, len(ray_dirs)))
      voxel_points.extend(list(map(lambda r: r * view_port_range + screen_center, [i/num_voxel_samples for i in range(0,num_voxel_samples)])))
  view_angles = list(map(lambda x: convertRayDirToAngles(x), ray_dirs))
  view_angles = list(map(lambda x: [x]* num_voxel_samples, view_angles))
  x = get_position_inputs(W, H)
  x = x.repeat(1, num_voxel_samples).reshape(len(x)*num_voxel_samples, -1)
  return torch.cat((x, torch.Tensor(voxel_points, requires_grad = False), torch.Tensor(view_angles, requires_grad = False).flatten(0,1)), 1)

# sample 3D point along ray (view direction), from near to far.
def sampleAlongViewDir(view_dir, screen_point, n_samples, view_port_range = 360):
    return torch.linspace(screen_point, screen_point + view_dir * view_port_range , n_samples)

num_voxel_samples = 10
input0 = getInputFromCameraPosition(0, W, H, view_port_range = W, num_voxel_samples = num_voxel_samples)  # view angle for image
input1 = getInputFromCameraPosition(-math.pi/4, W, H, view_port_range = W, num_voxel_samples = num_voxel_samples) # view angle for image 1
inputs = torch.cat((input0, input1), 0)
del input0, input1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

target_image = torch.Tensor(np.asarray(image).flatten())
target_image1 = torch.Tensor(np.asarray(image1).flatten())
target_image = torch.cat((target_image, target_image1), 0).to(device)
del target_image1

# Decrease learning rate if velocity is slow and std is high
# Increase learning rate if velocity is slow and std is low
# Stop if learning negatively

lossTracker = hanz_util.LossTracker()

mini_batch_size = 129600
train_dataloader = DataLoader(inputs, batch_size = mini_batch_size * num_voxel_samples, shuffle = True)
test_dataloader = DataLoader(target_image, batch_size = mini_batch_size, shuffle = True)
train_data_iter = iter(train_dataloader)
test_data_iter = iter(test_dataloader)

num_training_rounds = 10 * 2
for i in range(1,num_training_rounds):
    inputs = next(train_data_iter, None)
    if inputs == None:
        print("Reloading iterators..")
        train_data_iter = iter(train_dataloader)
        test_data_iter = iter(test_dataloader)
        inputs = next(train_data_iter, None)
        continue

    inputs = inputs.to(device)
    voxel_result = model(inputs)
    del inputs
    result_image = convertVoxelRaysToResult(voxel_result, num_voxel_samples = num_voxel_samples)
    del voxel_result
    target_image = next(test_data_iter).to(device)
    loss = torch.nn.L1Loss()(result_image.flatten(), target_image)
    print("{}/{} loss".format(i, num_training_rounds), loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lossTracker.push(loss)

# Input: lists of N voxels rgb colors along a ray, each row is a list of colors from near to far
#       dimension: (N * num_voxel_samples, 4)
# Output: (N, 3) colors
def convertVoxelRaysToResult(voxels_along_ray, num_voxel_samples = 7):
    rays = voxels_along_ray.reshape(-1, num_voxel_samples, 4)
    alphas = torch.cumprod(1 - rays[:,:, 3], 1).reshape(-1, 1).repeat(1,3)
    rgbs = rays[:,:, 0:3].reshape(-1,3)
    return alphas * rgbs

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


# Generate images by using intermediate angles
start_angle = 0
end_angle = -np.pi/4
steps = 3
angles = torch.linspace(start_angle, end_angle, steps)
for angle in angles.numpy():
  input = getInputFromCameraPosition(angle, W, H, view_port_range = W)
  result = model(input)
  img = convertResultToImage(result, W, H)
  showImage(img)
