import numpy as np
from physics_sim import PhysicsSim

class HoverTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        print(init_pose)
        print(self.target_pos)

    # x2
    def get_reward(self):
        distance = (self.sim.pose[:3] - self.target_pos) * [0.3, 0.3, 1]  # emphasizes the z-distance
        distance = 1 / (np.linalg.norm(distance) + 1)

        #velocity_z = np.tanh(self.sim.v[2]) * 2 # -0.5 ~ 0.5
        #reward = 0.8 * distance + 0.2 * velocity_z

        reward = 2 * distance - 1
        return reward / self.action_repeat, -1, 1

    # distance + z_velocity
    def get_reward_distance_vz(self):
        # x,y,Z distance
        distance = (self.sim.pose[:3] - self.target_pos) * [0.2, 0.2, 1]  # emphasizes the z-distance
        distance = 1 / (np.linalg.norm(distance) + 1) # 0~1.0
        distance = 2 * distance - 1.0 # -1~1.0

        # z velocity to be (+)
        v_z = 2 * np.tanh(self.sim.v[2]) # -1~1

        # sum
        reward = 0.8 * distance + 0.2 + v_z

        return reward / self.action_repeat


    def get_reward_x1(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        distance = (self.sim.pose[:3] - self.target_pos) * [0.2, 0.2, 1]  # emphasizes the z-distance
        distance = 1 / (np.linalg.norm(distance) + 1)
        reward = 2 * distance - 1.0

        reward += np.tanh(self.sim.v[2])
        reward = np.clip(reward, -1, 1)
        return reward / self.action_repeat

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state