#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.generated.const import GEOM_SPHERE
import glob
from matplotlib import cm

model = load_model_from_path("/home/gym/gym/envs/robotics/assets/fetch/push.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
index = 500_000

path = "/media/Codes/outpace/outpace_analyses/hgg_analysis/far_away_from_me/log"
index_number = 17

intermediate_goals_total = []
for i in range(index_number,index_number+1):
    intermediate_goals = np.load(path + "/env_goals" + str(i) + ".npy", allow_pickle=True)
    intermediate_goals = intermediate_goals.reshape(-1,3)
    intermediate_goals_total.append(intermediate_goals)


intermediate_goals = np.array(intermediate_goals_total).reshape(-1,3)
print(intermediate_goals)

robot_trajectory = np.load(path + "/test_obs" + str(index_number) + ".npy")


colors = cm.gist_rainbow(range(intermediate_goals.shape[0] * 2))

while True:
    for i in range(0, intermediate_goals.shape[0], 1):
        viewer.add_marker(type=GEOM_SPHERE,
                          pos=np.asarray(list(intermediate_goals[i,0:3])),
                          rgba=colors[i* 2],
                          size=np.asarray(([0.01] * 3)),
                          label=""
                          )
        
    for i in range(0, robot_trajectory.shape[0], 20):
        for j in range(0, robot_trajectory.shape[1], 2):
            viewer.add_marker(type=GEOM_SPHERE,
                            pos=np.asarray(list(robot_trajectory[i,j,:3])),
                            rgba=[1,0,0,1],
                            size=np.asarray(([0.01] * 3)),
                            label=""
                            )
            viewer.add_marker(type=GEOM_SPHERE,
                            pos=np.asarray(list(robot_trajectory[i,j,3:6])),
                            rgba=[0,1,0,1],
                            size=np.asarray(([0.03] * 3)),
                            label=""
                            )
    tmp = viewer.render()