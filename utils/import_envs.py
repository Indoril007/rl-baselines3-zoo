import gym
from gym.envs.registration import register

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

