# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License

"""A pointmass maze env.

Ported from legacy mujoco-py / d4rl to modern ``mujoco`` (>=3.x).  The
``OfflineEnv`` base class from d4rl and the ``gym.envs.mujoco.MujocoEnv``
class (which required mujoco-py) are replaced with a standalone
``gym.Env`` that uses ``mujoco.MjModel`` / ``mujoco.MjData`` directly.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gym
import mujoco
import numpy as np
from gym import spaces, utils

from .dynamic_mjc import MJCModel

WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split("\\")
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == "#":
                maze_arr[w][h] = WALL
            elif tile == "G":
                maze_arr[w][h] = GOAL
            elif tile == " " or tile == "O" or tile == "0":
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError("Unknown tile type: %s" % tile)
    return maze_arr


def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel("point_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        type="2d",
        name="groundplane",
        builtin="checker",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        width=100,
        height=100,
    )
    asset.texture(
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=".4 .6 .8",
        rgb2="0 0 0",
        width="800",
        height="800",
        mark="random",
        markrgb="1 1 1",
    )
    asset.material(name="groundplane", texture="groundplane", texrepeat="20 20")
    asset.material(name="wall", rgba=".7 .5 .3 1")
    asset.material(name="target", rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4", diffuse=".8 .8 .8", specular="0.1 0.1 0.1")
    visual.map(znear=0.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(
        name="ground",
        size="40 40 0.25",
        pos="0 0 -0.1",
        type="plane",
        contype=1,
        conaffinity=0,
        material="groundplane",
    )

    particle = worldbody.body(name="particle", pos=[1.2, 1.2, 0])
    particle.geom(
        name="particle_geom", type="sphere", size=0.1, rgba="0.0 0.0 1.0 0.0", contype=1
    )
    particle.site(
        name="particle_site", pos=[0.0, 0.0, 0], size=0.2, rgba="0.3 0.6 0.3 1"
    )
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    worldbody.site(name="target_site", pos=[0.0, 0.0, 0], size=0.2, material="target")

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w, h] == WALL:
                worldbody.geom(
                    conaffinity=1,
                    type="box",
                    name="wall_%d_%d" % (w, h),
                    material="wall",
                    pos=[w + 1.0, h + 1.0, 0],
                    size=[0.5, 0.5, 0.2],
                )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = (
    "############\\"
    + "#OOOO#OOOOO#\\"
    + "#O##O#O#O#O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O####O###O#\\"
    + "#OO#O#OOOOO#\\"
    + "##O#O#O#O###\\"
    + "#OO#OOO#OGO#\\"
    + "############"
)

LARGE_MAZE_EVAL = (
    "############\\"
    + "#OO#OOO#OGO#\\"
    + "##O###O#O#O#\\"
    + "#OO#O#OOOOO#\\"
    + "#O##O#OO##O#\\"
    + "#OOOOOO#OOO#\\"
    + "#O##O#O#O###\\"
    + "#OOOO#OOOOO#\\"
    + "############"
)

MEDIUM_MAZE = (
    "########\\"
    + "#OO##OO#\\"
    + "#OO#OOO#\\"
    + "##OOO###\\"
    + "#OO#OOO#\\"
    + "#O#OO#O#\\"
    + "#OOO#OG#\\"
    + "########"
)

MEDIUM_MAZE_EVAL = (
    "########\\"
    + "#OOOOOG#\\"
    + "#O#O##O#\\"
    + "#OOOO#O#\\"
    + "###OO###\\"
    + "#OOOOOO#\\"
    + "#OO##OO#\\"
    + "########"
)

SMALL_MAZE = "######\\" + "#OOOO#\\" + "#O##O#\\" + "#OOOO#\\" + "######"

U_MAZE = "#####\\" + "#GOO#\\" + "###O#\\" + "#OOO#\\" + "#####"

U_MAZE_EVAL = "#####\\" + "#OOG#\\" + "#O###\\" + "#OOO#\\" + "#####"

OPEN = "#######\\" + "#OOOOO#\\" + "#OOGOO#\\" + "#OOOOO#\\" + "#######"


STATE_RANGES = np.array(
    [
        [0.39318362, 3.2198412],  # Range for first dimension
        [0.62660956, 3.2187355],  # Range for second dimension
        [-5.2262554, 5.2262554],  # Range for third dimension
        [-5.2262554, 5.2262554],  # Range for fourth dimension
        [0.90001136, 3.0999563],  # Range for first dimension of target
        [0.9000267, 3.0999668],  # Range for second dimension of target
    ]
)

OFF_TARGET = np.array([10, 10])


class MazeEnv(gym.Env, utils.EzPickle):
    """PointMaze environment using modern ``mujoco`` (>=3.x).

    Replaces the legacy ``mujoco_env.MujocoEnv`` + ``d4rl.offline_env.OfflineEnv``
    base classes with a standalone ``gym.Env`` implementation.

    Args:
        maze_spec: Maze layout string (e.g. ``U_MAZE``).
        reward_type: ``"dense"`` or ``"sparse"``.
        reset_target: Whether to randomize the goal on each reset.
        return_value: ``"obs"`` for pixel observations, ``"state"`` for state.
        with_target: Whether to append target position to the state.
        **kwargs: Legacy ``OfflineEnv`` kwargs (``dataset_url``, ``ref_min_score``,
            ``ref_max_score``), stored but unused.
    """

    def __init__(
        self,
        maze_spec=U_MAZE,
        reward_type="dense",
        reset_target=True,
        return_value="state",
        with_target=False,
        **kwargs,
    ):
        self.dataset_url = kwargs.get("dataset_url", None)
        self.ref_min_score = kwargs.get("ref_min_score", None)
        self.ref_max_score = kwargs.get("ref_max_score", None)

        self.with_target = with_target
        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0, 0.0])
        self.return_value = return_value
        self.frame_skip = 5

        model = point_maze(maze_spec)
        with model.asfile() as f:
            self.mj_model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.data)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        obs_size = self.mj_model.nq + self.mj_model.nv  # qpos + qvel
        obs_high = np.inf * np.ones(obs_size)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.mj_model.nu,), dtype=np.float32
        )

        utils.EzPickle.__init__(self)

        if self.return_value == "obs":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
            )

        self._renderer = None

        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            self.set_target(
                np.array(self.reset_locations[0]).astype(self.observation_space.dtype)
            )
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

        self.seed()
        self.reset_to_state = None

    def _get_renderer(self, width=224, height=224):
        """Lazily create or recreate the offscreen renderer."""
        if (
            self._renderer is None
            or self._render_w != width
            or self._render_h != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
            self._render_w = width
            self._render_h = height
        return self._renderer

    def do_simulation(self, action, n_frames):
        self.data.ctrl[:] = action
        for _ in range(n_frames):
            mujoco.mj_step(self.mj_model, self.data)

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.mj_model, self.data)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        state = self._get_obs()["proprio"]
        state = np.concatenate([state, self._target]) if self.with_target else state
        if self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(state[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == "dense":
            reward = np.exp(-np.linalg.norm(state[0:2] - self._target))
        else:
            raise ValueError("Unknown reward type %s" % self.reward_type)
        done = False
        if self.return_value == "obs":
            visual = self._render_frame()
            ob = {
                "visual": visual,
                "proprio": state,
            }

        else:
            ob = {
                "visual": state,
                "proprio": state,
            }

        info = {}
        info["state"] = state
        info["target"] = self._target
        info["obs"] = ob if self.return_value == "obs" else None
        info["pos_agent"] = state[:2]
        return ob, reward, done, info

    def _get_obs(self):
        obs = {
            "visual": np.concatenate([self.data.qpos, self.data.qvel])
            .ravel()
            .astype(np.float32),
            "proprio": np.concatenate([self.data.qpos, self.data.qvel])
            .ravel()
            .astype(np.float32),
        }
        return obs

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(
                self.observation_space.dtype
            )
            target_location = reset_location + self.np_random.uniform(
                low=-0.1, high=0.1, size=self.mj_model.nq
            )
        self._target = target_location

    def set_marker(self):
        if not self.with_target:
            self.set_target(OFF_TARGET)
        site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self.data.site_xpos[site_id] = np.array(
            [self._target[0] + 1, self._target[1] + 1, 0.0]
        )

    def clip_velocity(self):
        qvel = np.clip(self.data.qvel, -5.0, 5.0)
        self.set_state(self.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(
            self.observation_space.dtype
        )
        qpos = reset_location + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.mj_model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.mj_model.nv) * 0.1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        mujoco.mj_resetData(self.mj_model, self.data)
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.mj_model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.mj_model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass

    def set_init_state(self, init_state):
        self.reset_to_state = init_state
        target = init_state[-2:] if init_state is not None else None
        self.set_target(target)
        self.set_marker()

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.data)
        self.set_init_state(self.reset_to_state)
        state = self.reset_to_state
        if state is None:
            rs = self.random_state
            state = np.array(
                [
                    rs.uniform(low=STATE_RANGES[0][0], high=STATE_RANGES[0][1]),
                    rs.uniform(low=STATE_RANGES[1][0], high=STATE_RANGES[1][1]),
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                    rs.uniform(low=STATE_RANGES[4][0], high=STATE_RANGES[4][1]),
                    rs.uniform(low=STATE_RANGES[5][0], high=STATE_RANGES[5][1]),
                ]
            )
        qpos, qvel = state[:2], state[2:4]
        self.set_state(qpos, qvel)
        self.set_marker()
        obs = self._get_obs()
        if self.return_value == "obs":
            visual = self._render_frame()
            obs["visual"] = visual
        state = state[:4] if self.with_target else state
        return obs, state

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)

    def prepare_for_render(self):
        self.return_value = "obs"
        init_state = np.array([1.0856, 1.9746, 0.0098, 0.0217])
        self.set_state(init_state[:2], init_state[2:])

        cam = mujoco.MjvCamera()
        cam.azimuth = 270
        cam.elevation = -90
        cam.distance = 8.0
        cam.lookat[:] = [3.0, 3.0, 0.0]
        self._cam = cam

    def _render_frame(self, width=224, height=224):
        renderer = self._get_renderer(width, height)
        if hasattr(self, "_cam"):
            renderer.update_scene(self.data, camera=self._cam)
        else:
            renderer.update_scene(self.data)
        return renderer.render()

    def render_at_position(self, pos, width=224, height=224) -> np.ndarray:
        """Render observation with particle at ``pos`` without modifying env state.

        Args:
            pos: 2D position (numpy array or tensor) in MuJoCo coordinates.
            width: Render width.
            height: Render height.

        Returns:
            ``[H, W, 3]`` uint8 numpy array.
        """
        if hasattr(pos, "numpy"):
            pos = pos.numpy()
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.data.qpos[:] = pos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.data)
        img = self._render_frame(width, height)
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.mj_model, self.data)
        return img
