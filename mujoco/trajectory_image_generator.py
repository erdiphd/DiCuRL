import numpy as np
import os
from os.path import join as ospj
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import matplotlib.cm as cm
import cv2
import subprocess

from mpl_toolkits.mplot3d import Axes3D




def plot2d(path, start, end, step=1):
    for i in range(start,end,step):
        observation = np.load(path + "/test_obs" + str(i)+".npy")
        intermediate_goals = np.load(path + "/env_goals" + str(i) + ".npy", allow_pickle=True)

        achieved_trajectory = observation[:,:,0:3].reshape(-1,3)
        desired_goal = observation[:,:,-3:].reshape(-1,3)
        intermediate_goals = intermediate_goals[:,:]
        plt.nipy_spectral()
        fig, ax = plt.subplots()
        plt.scatter(x =achieved_trajectory[:,0], y = achieved_trajectory[:,1], s=5, c=range(achieved_trajectory.shape[0]), cmap=cm.gist_rainbow)
        plt.scatter(x =desired_goal[:,0], y = desired_goal[:,1], s=100, color = 'red')
        plt.scatter(x =intermediate_goals[:,0], y = intermediate_goals[:,1], s=5, color = 'green')
        plt.colorbar()
        ax.set_xlim(0.5, 2.0)  # Set x-axis limits from 0 to 8
        ax.set_ylim(0.1, 1.6)  #
        plt.title("Timestep:" + str(i))
        plt.savefig("images/curriculum_goal" + str(i) + ".png")


def plot3d(path, start, end, step=1):
    for i in range(start, end, step):
        observation = np.load(path + "/test_obs" + str(i) + ".npy")
        intermediate_goals = np.load(path + "/env_goals" + str(i) + ".npy", allow_pickle=True)

        achieved_trajectory = observation[:, :, 0:3].reshape(-1, 3)
        desired_goal = observation[:, :, -3:].reshape(-1, 3)
        intermediate_goals = intermediate_goals[:, :]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(achieved_trajectory[:, 0], achieved_trajectory[:, 1], achieved_trajectory[:, 2], s=5, c=range(achieved_trajectory.shape[0]), cmap='gist_rainbow')
        ax.scatter(desired_goal[:, 0], desired_goal[:, 1], desired_goal[:, 2], s=30, color='red')
        ax.scatter(intermediate_goals[:, 3], intermediate_goals[:, 4], intermediate_goals[:, 5], s=5, color='green')

        ax.set_xlim(0.5, 2.0)
        ax.set_ylim(0.1, 1.6)
        ax.set_zlim(0.4, 0.9)  # Assuming the z-axis limits, you can adjust this based on your data

        ax.set_title("Timestep:" + str(i))

        plt.savefig("images/curriculum_goal" + str(i) + ".png")
        plt.close()  # Close the figure to avoid opening multiple figures



def video_generator():
    image_folder = 'images'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images, key=lambda images: int(images[15:images.find(".")]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 50, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    input_file = video_name
    output_file = 'video.mp4'

    # H.264 example
    command = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-strict', 'experimental', '-b:a', '192k', output_file]
    subprocess.run(command)
    subprocess.run(['rm', '-rf', input_file])

# video_generator()


path = "/media/Codes/outpace/outpace_analyses/hgg_analysis/tmp/"
plot2d(path,1,85)
# video_generator()
