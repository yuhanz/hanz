from PIL import Image
import matplotlib.pyplot as plt

import torch
import hanz
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

def rotateCameraAroundOrigin(angle):
  camera_distance = -360
  screen_distance = -180
  cosA = np.cos(angle)
  sinA = np.sin(angle)
  camera_pos = np.array([sinA, 0, cosA]) * camera_distance
  screen_center = np.array([sinA, 0, cosA]) * screen_distance
  return [camera_pos, screen_center]


def getInputFromCameraPosition(angle, screen_width, screen_height):
  W = screen_width
  H = screen_height
  get_position_inputs(W, H)
  camera_pos, screen_center = rotateCameraAroundOrigin(angle)
  ray_dirs = torch.Tensor(getViewDirs(camera_pos, screen_center, W, H))
  ray_origins = torch.Tensor([camera_pos] * (W * H))
  x = get_position_inputs(W, H)
  return torch.cat((x, ray_origins, ray_dirs), 1)

input0 = getInputFromCameraPosition(0, W, H)  # view angle for image
input1 = getInputFromCameraPosition(-math.pi/4, W, H) # view angle for image 1
inputs = torch.cat((input0, input1), 0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


learning_rate = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

target_image = torch.Tensor(np.asarray(image).flatten())
target_image1 = torch.Tensor(np.asarray(image1).flatten())
target_image = torch.cat((target_image, target_image1), 0).to(device)

# Decrease learning rate if velocity is slow and std is high
# Increase learning rate if velocity is slow and std is low
# Stop if learning negatively
class LossTracker:
    lossMean = None
    lossStd = None
    lossSamples = []
    lossSamplesToCollect = 20
    lossVelocity = 0
    start_timestamp = None
    def push(self, loss):
        self.lossSamples.append(loss.item())
        if(self.lossMean == None):
            self.lossMean = self.lossSamples[0]
            self.start_timestamp = time.time()
        if len(self.lossSamples) >= self.lossSamplesToCollect:
            lossMean2 = np.mean(self.lossSamples)
            lossStd2 = np.std(self.lossSamples)
            self.lossVelocity = lossMean2 - self.lossMean
            self.lossMean = lossMean2
            self.lossStd = lossStd2
            self.lossSamples = []
            self.display()
            self.start_timestamp = time.time()
    def display(self):
        t = time.time() - self.start_timestamp
        rate_changed = self.lossVelocity / self.lossMean
        target_rate_change = 0.1
        ratio = abs(target_rate_change / rate_changed)
        seconds = int(ratio * t)
        steps = int(ratio * self.lossSamplesToCollect)
        print("-- loss velocity in {} steps: {}".format(self.lossSamplesToCollect, self.lossVelocity))
        print("----- {} {}% in {} seconds ({} steps)".format('Reduce' if self.lossVelocity<0 else 'Increase',  target_rate_change*100, seconds, steps))

lossTracker = LossTracker()

pdb.set_trace()

num_training_rounds = 2
inputs = inputs.to(device)
for i in range(1,num_training_rounds):
    result_image = model(inputs)
    loss = torch.nn.L1Loss()(result_image.flatten(), target_image )
    print("{}/{} loss".format(i, num_training_rounds), loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lossTracker.push(loss)

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
  input = getInputFromCameraPosition(angle, W, H)
  result = model(input)
  img = convertResultToImage(result, W, H)
  showImage(img)
