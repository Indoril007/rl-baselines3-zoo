import gym
from gym.envs.registration import register
import numpy as np

from utils.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

try:
    import panda_gym  # pytype: disable=import-error
except ImportError:
    panda_gym = None

try:
    import rocket_lander_gym  # pytype: disable=import-error
except ImportError:
    rocket_lander_gym = None

try:
    import d4rl  # pytype: disable=import-error
    from gym.envs.registration import register
    from d4rl.locomotion import maze_env

    class StaticStartMazeEnv(d4rl.pointmaze.MazeEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reset_locations = [(1,1)]

        def reset_model(self):
            idx = self.np_random.choice(len(self.reset_locations))
            reset_location = np.array(self.reset_locations[idx]).astype(self.observation_space.dtype)
            qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            self.set_state(qpos, qvel)
            if self.reset_target:
                self.set_target()
            return self._get_obs()


    register(
        id='antmaze-umaze-v3',
        entry_point='d4rl.locomotion.ant:make_ant_maze_env',
        max_episode_steps=1400,
        kwargs={
            'deprecated': True,
            'maze_map': maze_env.U_MAZE_TEST,
            'reward_type':'sparse',
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_new/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse.hdf5',
            'non_zero_reset':False,
            'eval':True,
            'maze_size_scaling': 4.0,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0,
        }
    )

    register(
        id="maze2d-large-staticstart-v1",
        entry_point=StaticStartMazeEnv,
        max_episode_steps=800,
        kwargs={
            "maze_spec": d4rl.pointmaze.LARGE_MAZE,
            "reward_type": "sparse",
            "reset_target": False,
            "ref_min_score": 6.7,
            "ref_max_score": 273.99,
            "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5",
        },
    )

    register(
        id="maze2d-large-dense-staticstart-v1",
        entry_point=StaticStartMazeEnv,
        max_episode_steps=800,
        kwargs={
            "maze_spec": d4rl.pointmaze.LARGE_MAZE,
            "reward_type": "dense",
            "reset_target": False,
            'ref_min_score': 30.569041,
            'ref_max_score': 303.4857382709002,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-dense-v1.hdf5'
        },
    )


except ImportError:
    d4rl_env = None

# Register no vel envs
def create_no_vel_env(env_id: str):
    def make_env():
        env = gym.make(env_id)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),
    )

