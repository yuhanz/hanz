from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import hanz
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
    frequencies_x = torch.Tensor(np.array(list(map(lambda f: f * xs.numpy(), frequencies))))
    frequencies_y = torch.Tensor(np.array(list(map(lambda f: f * ys.numpy(), frequencies))))
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
    d = np.linalg.norm(ray_dir[0:2]) or 1.0
    a0 = ray_dir[0] / d
    a1 = ray_dir[1] / d
    a2 = ray_dir[2] / d
    if math.isnan(a0) or math.isnan(a1) or math.isnan(a2):
        print("---- nan from ray:", ray_dir)
    return [a0,a1,a2]

def getInputFromCameraPosition(angle, screen_width, screen_height, view_port_range, num_voxel_samples = 7, screen_distance = 180):
  W = screen_width
  H = screen_height
  camera_pos, screen_center = rotateCameraAroundOrigin(angle, screen_distance = 180)
  ray_dirs = getViewDirs(camera_pos, screen_center, W, H)
  voxel_points = []
  i = 0
  for ray_dir in ray_dirs:
      i = i +1
      print('---{}/{}'.format(i, len(ray_dirs)))
      voxel_points.extend(list(map(lambda r: r * view_port_range * ray_dir + screen_center, [i/num_voxel_samples for i in range(0,num_voxel_samples)])))
  view_angles = list(map(lambda x: convertRayDirToAngles(x), ray_dirs))
  view_angles = list(map(lambda x: [x]* num_voxel_samples, view_angles))
  x = get_position_inputs(W, H)
  x = x.repeat(1, num_voxel_samples).reshape(len(x)*num_voxel_samples, -1)
  return torch.cat((x, torch.Tensor(voxel_points), torch.Tensor(view_angles).flatten(0,1)), 1)

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

# Input: lists of N voxels rgb colors along a ray, each row is a list of colors from near to far
#       dimension: (N * num_voxel_samples, 4)
# Output: (N, 3) colors
def convertVoxelRaysToResult(voxels_along_ray, num_voxel_samples = 7):
    rays = voxels_along_ray.reshape(-1, num_voxel_samples, 4)
    alphas = torch.cumprod(1 - torch.nn.ReLU()(rays[:,:, 3]), 1).reshape(-1, 1).repeat(1,3)
    rgbs = rays[:,:, 0:3].reshape(-1,3)
    return torch.sum((alphas * rgbs).reshape(-1, num_voxel_samples, 3), 1)

# Tensor as input
def convertResultToImage(result_array, width, height):
    result_array = result_array.cpu().detach().numpy().astype('uint8')
    return convertNumpyResultToImage(result_array, width, height)

# numpy array as input
def convertNumpyResultToImage(result_array, width, height):
    generated_img = np.reshape(result_array, (width, height, 3))
    return generated_img

def showImage(img):
    fig,ax = plt.subplots()
    ax.imshow(img)
    plt.show()

# Decrease learning rate if velocity is slow and std is high
# Increase learning rate if velocity is slow and std is low
# Stop if learning negatively

lossTracker = hanz_util.LossTracker()

mini_batch_size = 12960
train_dataloader = DataLoader(inputs, batch_size = mini_batch_size * num_voxel_samples, shuffle = False)
test_dataloader = DataLoader(target_image, batch_size = mini_batch_size * 3, shuffle = False)
train_data_iter = iter(train_dataloader)
test_data_iter = iter(test_dataloader)

num_training_rounds = 110 * 10

for i in range(1,num_training_rounds+1):
    batch_inputs = next(train_data_iter, None)
    expected_target_image = next(test_data_iter, None)
    if batch_inputs == None:
        print("Reloading iterators..")
        train_data_iter = iter(train_dataloader)
        test_data_iter = iter(test_dataloader)
        batch_inputs = next(train_data_iter, None)
        expected_target_image = next(test_data_iter, None)
    batch_inputs = batch_inputs.to(device)
    voxel_result = model(batch_inputs)
    del batch_inputs
    result_image = convertVoxelRaysToResult(voxel_result, num_voxel_samples)
    del voxel_result
    expected_target_image = expected_target_image.to(device)
    loss = torch.nn.L1Loss()(result_image.flatten(), expected_target_image)
    print("{}/{} loss".format(i, num_training_rounds), loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # lossTracker.push(loss)

img = convertResultToImage(result_image[0:W*H], W, H)
img2 = convertResultToImage(result_image[W*H:], W, H)
showImage(img)
showImage(img2)

def getOutputWithBatch(model, input, mini_batch_size, device, num_voxel_samples):
    input_dataloader = DataLoader(input, batch_size = mini_batch_size * num_voxel_samples, shuffle = False)
    iterator = iter(input_dataloader)
    image_parts = []
    for batch_inputs in iterator:
        batch_inputs = batch_inputs.to(device)
        # voxel_result = model.cpu()(batch_inputs)
        voxel_result = model(batch_inputs)
        del batch_inputs
        result_image = convertVoxelRaysToResult(voxel_result, num_voxel_samples)
        del voxel_result
        image_parts.append(result_image.cpu().detach().numpy())
        del result_image
    return np.array(image_parts).flatten()


# Generate images by using intermediate angles
start_angle = 0
end_angle = -np.pi/4
steps = 7
angles = torch.linspace(start_angle, end_angle, steps)
for angle in angles.numpy():
  input = getInputFromCameraPosition(angle, W, H, view_port_range = W, num_voxel_samples = 10)
  # result = model(input)
  result = getOutputWithBatch(model, input, mini_batch_size = 12960, device = device, num_voxel_samples = num_voxel_samples)
  img = convertNumpyResultToImage(result.astype('uint8'), W, H)
  showImage(img)
