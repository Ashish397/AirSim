import gym
import math.
import time
import numpy as np
import airsim
import setup_path

from PIL import Image
from gym import spaces
from argparse import ArgumentParser
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, agents):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.agents = agents
        
        self.state = {}
        for agent in self.agents:
            self.state[agent] = {
                "position": np.zeros(3),
                "collision": False,
                "prev_position": np.zeros(3),
                "collision": None,
            }
        
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True, "Drone1")
        self.drone.enableApiControl(True, "Drone2")
        self.drone.armDisarm(True, "Drone1")
        self.drone.armDisarm(True, "Drone2")

        # Set home position and velocity
        self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10, vehicle_name="Drone1")
        self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5, vehicle_name="Drone2")

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        image = []
        for agent in self.agents:
            responses = self.drone.simGetImages([self.image_request], vehicle_name=agent)
            image.append(self.transform_obs(responses))
            self.drone_state = self.drone.getMultirotorState(vehicle_name=agent)

            self.state[agent]["prev_position"] = self.state[agent]["position"]
            self.state[agent]["position"] = self.drone_state.kinematics_estimated.position
            self.state[agent]["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

            collision = self.drone.simGetCollisionInfo(vehicle_name=agent).has_collided
            self.state[agent]["collision"] = collision
            
            print(
                str(self.state[agent]["position"].x_val)[0:4] + " // " +
                str(self.state[agent]["position"].y_val)[0:4] + " // " +
                str(self.state[agent]["position"].z_val)[0:4]
            )

        return image

    def _do_action(self, action):
        for i, agent in enumerate(self.agents):
            quad_offset = self.interpret_action(action[i])
            quad_vel = self.drone.getMultirotorState(vehicle_name=agent).kinematics_estimated.linear_velocity
            self.drone.moveByVelocityAsync(
                quad_vel.x_val + quad_offset[0],
                quad_vel.y_val + quad_offset[1],
                quad_vel.z_val + quad_offset[2],
                5,
                vehicle_name=agent,
            )

    def _compute_reward(self):
        thresh_dist = 7
        beta = 1
        
        pts = [
            np.array([50, 50, -20]),
        ]

        full_reward = [0] * len(self.agents)
        done = [0] * len(self.agents)
        
        for agent_num, agent in enumerate(self.agents):
            quad_pt = np.array(
                list(
                    (
                        self.state[agent]["position"].x_val,
                        self.state[agent]["position"].y_val,
                        self.state[agent]["position"].z_val,
                    )
                )
            )
            if self.state[agent]["collision"]:
                reward = -100
                done[agent_num] = [1]
            else:
                dist = 10000000
                dist = min(
                    dist,
                    np.linalg.norm(np.cross((quad_pt - pts[0]), (quad_pt - pts[1])))
                    / np.linalg.norm(pts[0] - pts[1]),
                )

            if dist > thresh_dist:
                reward = -10
            else:
                reward_dist = math.exp(-beta * dist) - 0.5
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )
                reward = reward_dist + reward_speed
            
            full_reward[agent_num] = reward
            done[agent_num] = 0
            if reward <= -10:
                done[agent_num] = [1]

        return reward, int(bool(sum(done)))

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self._get_obs()
        return self.state

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset

    def exit(self):
        self.drone.armDisarm(False, "Drone1")
        self.drone.armDisarm(False, "Drone2")
        
        self.drone.reset()
        
        self.drone.enableApiControl(False, "Drone1")
        self.drone.enableApiControl(False, "Drone2")
