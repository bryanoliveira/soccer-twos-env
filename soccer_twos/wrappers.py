from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import gym
from gym import spaces
from gym_unity.envs import UnityToGymWrapper, UnityGymException, ActionFlattener
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)
GymStepResult = Tuple[np.ndarray, float, bool, Dict]


class EnvType(Enum):
    multiagent_player = "multiagent_player"
    multiagent_team = "multiagent_team"
    team_vs_policy = "team_vs_policy"


class TerminationMode:
    ALL = "ALL"
    ANY = "ANY"
    MAJORITY = "MAJORITY"


class MultiAgentUnityWrapper(UnityToGymWrapper):
    """An implementation of the UnityToGymWrapper that supports multi-agent environments.
    Based on `UnityToGymWrapper` from the Unity's ML-Toolkits [1] and on ai-traineree modifications [2]:
    Updated to work with ml-agents v0.27.0.

    At the time of writting the official package doesn't support multi agents.
    Until it's clear why it doesn't support [3] and whether they plan on adding
    anything, we're keeping this version. When the fog of unknown has been
    blown away, we might consider doing a Pull Request to `ml-agents`.

    [1]: https://github.com/Unity-Technologies/ml-agents/blob/56e6d333a52863785e20c34d89faadf0a115d320/gym-unity/gym_unity/envs/__init__.py
    [2]: https://github.com/laszukdawid/ai-traineree/blob/a9d89b458e40724211d4a0cc8331886dead3eb57/ai_traineree/tasks.py#L260
    [3]: https://github.com/Unity-Technologies/ml-agents/issues/4120

    Here, agents are `members` distributed in `groups`. Members are actual agents,
    while groups are logical grouping of behaviors. For example, a team on "SoccerTwos"
    would be a team of two members (agents).
    """

    def __init__(
        self,
        unity_env: BaseEnv,
        allow_multiple_obs: bool = False,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        action_space_seed: Optional[int] = None,
        termination_mode: str = TerminationMode.ANY,
        soccer_twos_only: bool = True,
    ):
        """
        Environment initialization
        Args:
            unity_env: The Unity BaseEnv to be wrapped in the gym. Will be closed when the UnityToGymWrapper closes.
            uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
            flatten_branched: If True, turn branched discrete action spaces into a Discrete space rather than MultiDiscrete.
            allow_multiple_obs: Only here for backwards compatibility. Has no effect.
            action_space_seed: If non-None, will be used to set the random seed on created gym.Space instances.
            termination_mode: A string (enum) suggesting when to end an episode. Supports "ANY", "MAJORITY" and "ALL"
                which are atributes on `TerminationMode`.
        """
        self._env = unity_env

        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()

        self.visual_obs = None

        # Save the step result from the last time all Agents requested decisions.
        self._previous_decision_step: DecisionSteps = None
        self._flattener = None
        # Hidden flag used by Atari environments to determine if the game is over
        self.game_over = False

        # < multiagent mod > (removed len check)
        # When to stop the game, considering all agents
        assert termination_mode in TerminationMode.__dict__
        self.termination_mode = termination_mode

        self.name = list(self._env.behavior_specs.keys())[0]
        # < multiagent mod > (added agent_prefix)
        self.group_prefix = self.name[: self.name.index("=") + 1]
        self.group_spec = self._env.behavior_specs[self.name]

        if self._get_n_vis_obs() == 0 and self._get_vec_obs_size() == 0:
            raise UnityGymException(
                "There are no observations provided by the environment."
            )

        if not self._get_n_vis_obs() >= 1 and uint8_visual:
            logger.warning(
                "uint8_visual was set to true, but visual observations are not in use. "
                "This setting will not have any effect."
            )
        else:
            self.uint8_visual = uint8_visual

        # Check for number of agents in scene.
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.name)
        # < multiagent mod > (removed len check)
        self.num_groups = len(self._env.behavior_specs)
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            if self.group_spec.action_spec.discrete_size == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        elif self.group_spec.action_spec.is_continuous():
            if flatten_branched:
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )

            self.action_size = self.group_spec.action_spec.continuous_size
            high = np.array([1] * self.group_spec.action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            raise UnityGymException(
                "The gym wrapper does not provide explicit support for both discrete "
                "and continuous actions."
            )

        if action_space_seed is not None:
            self._action_space.seed(action_space_seed)

        # Set observations space
        # get visual obs
        list_spaces: List[gym.Space] = []
        self._has_vis_obs = False
        if self._get_n_vis_obs() > 0:
            self._has_vis_obs = True
            shapes = self._get_vis_obs_shape()
            for shape in shapes:
                if uint8_visual:
                    list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
                else:
                    list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        # get vector obs
        self._has_vec_obs = False
        if self._get_vec_obs_size() > 0:
            self._has_vec_obs = True
            vec_space = spaces.Box(
                -np.inf, np.inf, dtype=np.float32, shape=(self._get_vec_obs_size(),)
            )
            list_spaces.append(vec_space)

        self.soccer_twos_only = soccer_twos_only
        if soccer_twos_only:
            self._observation_space = spaces.Box(
                -np.inf,
                np.inf,
                dtype=np.float32,
                shape=(336,),  # hardcoded for compiled env
            )
        else:
            self._observation_space = (
                spaces.Tuple(list_spaces) if len(list_spaces) > 1 else list_spaces[0]
            )

    def reset(self) -> Union[Dict[int, np.ndarray], np.ndarray]:
        """Resets the state of the environment and returns an initial observation.
        If the number of agents is greater than one, the observations will be a dict.
        Returns:
            observation (object/list): the initial observation of the space.
        """
        self._env.reset()
        # < multiagent mod >
        if self.num_groups > 1:
            obs_dict = {}
            for group_id in range(self.num_groups):
                decision_step, _ = self._env.get_steps(
                    self.group_prefix + str(group_id)
                )
                self.game_over = False
                obs, *_ = self._single_step(decision_step)
                for member_id in obs:
                    obs_dict[group_id * len(obs) + member_id] = obs[member_id]
            return obs_dict
        else:
            decision_step, _ = self._env.get_steps(self.name)
            self.game_over = False
            obs, *_ = self._single_step(decision_step)
            return obs  # res contains tuple with `state` on first pos

    def step(self, action: Union[Dict[int, List[Any]], List[Any]]) -> GymStepResult:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object/list): a single action or a dict with actions for each agent (multiagent setting)
        Returns:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        if self.game_over:
            raise UnityGymException(
                "You are calling 'step()' even though this environment has already "
                "returned done = True. You must always call 'reset()' once you "
                "receive 'done = True'."
            )

        # < multiagent mod >
        if self.num_groups > 1:
            assert (
                type(action) is dict
            ), "The environment requires a dictionary for multi-agent setting."
            num_group_members = len(action) // self.num_groups
            for group_id in range(self.num_groups):
                group_actions = []
                for i in range(num_group_members):
                    group_actions.append(action[group_id * num_group_members + i])
                self._set_action(group_actions, self.group_prefix + str(group_id))
        else:
            self._set_action(action, self.name)

        self._env.step()

        # < multiagent mod >
        if type(action) is dict:
            obs_dict = {}
            rew_dict = {}
            info_dict = {}
            done = False  # done here is treated environment-wise
            for group_id in range(self.num_groups):
                o, r, d, i = self._get_step_results(self.group_prefix + str(group_id))
                for member_id in o:
                    obs_dict[group_id * len(o) + member_id] = o[member_id]
                    rew_dict[group_id * len(o) + member_id] = r[member_id]
                    info_dict[group_id * len(o) + member_id] = i[member_id]
                # done uses team-wise info
                done = done or d
            done_dict = {"__all__": done}
            return obs_dict, rew_dict, done_dict, info_dict
        else:
            return self._get_step_results(self.name)

    # < multiagent mod > (append all vector obs)
    def _single_step(self, info: Union[DecisionSteps, TerminalSteps]) -> GymStepResult:
        observations = []
        if self._has_vis_obs:
            # gather visual observations
            vis_obs = []
            for obss in self._get_vis_obs_list(info):
                vis_obs.append([self._preprocess_single(obs) for obs in obss])
            # frame from first agent (group 0, member 0), used in render()
            self.visual_obs = vis_obs[0][0]
            observations.append(vis_obs)
        if self._has_vec_obs:
            # gather vector observations
            vector_obs = self._get_vector_obs(info)
            observations.append(vector_obs)

        # create dicts like {member_id: info} for member_id in group_id
        observations = {
            i: (
                tuple(observations[0][i], observations[1][i])
                if len(observations) > 1
                else observations[0][i]
            )
            for i in range(len(observations[0]))
        }
        rewards = {
            i: info.reward[i] + info.group_reward[i] for i in range(len(info.reward))
        }
        done = isinstance(info, TerminalSteps)
        info = {}

        # specific for soccer-twos: ray observations, player info, ball info
        if self.soccer_twos_only:
            for member_id in observations:
                _obs = observations[member_id]
                observations[member_id] = _obs[:336]
                if len(_obs) == 345:
                    # binary env sent extra info
                    info[member_id] = {
                        "player_info": {
                            "position": _obs[336:338],
                            "rotation_y": _obs[338],
                            "velocity": _obs[339:341],
                        },
                        "ball_info": {
                            "position": _obs[341:343],
                            "velocity": _obs[343:345],
                        },
                    }

        return observations, rewards, done, info

    def _set_action(self, actions: List[Any], group_name: str) -> None:
        """Sets the actions for an group within the environment.
        Args:
            actions (list): the actions to take
            group_name (str): the name of the group to set the actions for
        """

        if self._flattener is not None:
            # unflatten actions
            actions = [self._flattener.lookup_action(action) for action in actions]

        actions = np.array(actions).reshape((-1, self.action_size))

        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(actions)
        else:
            action_tuple.add_discrete(actions)

        self._env.set_actions(group_name, action_tuple)

    def _get_step_results(self, group_name: str) -> GymStepResult:
        """Returns the observation, reward and whether the episode is over after taking the action.
        Args:
            group_name (str): the name of the group to get the step results for
        Returns:
            observation (object/list): group's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        decision_step, terminal_step = self._env.get_steps(group_name)

        if self._detect_game_over(terminal_step):
            self.game_over = True
            out = self._single_step(terminal_step)
            return out
        else:
            return self._single_step(decision_step)

    # < multiagent mod >
    def _detect_game_over(self, terminal_steps: List[TerminalSteps]) -> bool:
        """Determine whether the episode has finished.

        Expects the `terminal_steps` to contain only steps that terminated. Note that other steps
        are possible in the same iteration.
        This is to keep consistent with Unity's framework but likely will go through refactoring.

        Args:
            terminal_steps (list): list of all the steps that terminated.
        """
        return (
            (self.termination_mode == TerminationMode.ANY and len(terminal_steps) > 0)
            or (
                self.termination_mode == TerminationMode.MAJORITY
                and len(terminal_steps) > 0.5 * self.num_groups
            )
            or (
                self.termination_mode == TerminationMode.ALL
                and len(terminal_steps) == self.num_groups
            )
        )


class MultiagentTeamWrapper(gym.core.Wrapper):
    """
    A wrapper for multiagent team-controlled environment.
    Uses a 2x2 (4 players) environment to expose a 1x1 (2 teams) environment.
    """

    def __init__(self, env):
        super(MultiagentTeamWrapper, self).__init__(env)
        self.env = env
        # duplicate obs & action spaces (concatenate players)
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0] * 2,)
        )
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(env.action_space.n ** 2)
            self.action_space_n = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_space = gym.spaces.MultiDiscrete(
                np.repeat(env.action_space.nvec, 2)
            )
            self.action_space_n = len(env.action_space.nvec)
        else:
            raise ValueError("Unsupported action space type")

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            env_action = {
                # actions for team 1
                0: action[0] // self.action_space_n,
                1: action[0] % self.action_space_n,
                # actions for team 2
                2: action[1] // self.action_space_n,
                3: action[1] % self.action_space_n,
            }
        else:
            env_action = {
                # slice actions for team 1
                0: action[0][: self.action_space_n],
                1: action[0][self.action_space_n :],
                # slice actions for team 2
                2: action[1][: self.action_space_n],
                3: action[1][self.action_space_n :],
            }
        obs, reward, done, info = self.env.step(env_action)
        return (
            self._preprocess_obs(obs),
            self._preprocess_reward(reward),
            done,
            self._preprocess_info(info),
        )

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return {
            0: np.concatenate((obs[0], obs[1])),
            1: np.concatenate((obs[2], obs[3])),
        }

    def _preprocess_reward(self, reward):
        return {
            0: reward[0] + reward[1],
            1: reward[2] + reward[3],
        }

    def _preprocess_info(self, info):
        return {
            0: {0: info[0], 1: info[1]},
            1: {0: info[2], 1: info[3]},
        }


class TeamVsPolicyWrapper(gym.core.Wrapper):
    """
    A wrapper for team vs given policy environment.
    Uses random policy as opponent by default.
    Uses a 2x2 (4 players) environment to expose a 1x1 (2 teams) environment.
    """

    def __init__(
        self,
        env,
        opponent_policy: Callable = None,
        teammate_policy: Callable = None,
        single_player: bool = False,
    ):
        super(TeamVsPolicyWrapper, self).__init__(env)
        self.env = env
        self.single_player = single_player
        self.last_obs = None

        # duplicate obs & action spaces
        self.observation_space = gym.spaces.Box(
            0,
            1,
            dtype=np.float32,
            shape=(env.observation_space.shape[0] * (1 if single_player else 2),),
        )
        if isinstance(env.action_space, gym.spaces.Discrete):
            # every combination of actions for both players
            self.action_space = gym.spaces.Discrete(
                env.action_space.n ** (1 if single_player else 2)
            )
            self.action_space_n = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.single_player = False  # disable single_player when using MultiDiscrete
            self.action_space = gym.spaces.MultiDiscrete(
                np.repeat(env.action_space.nvec, 2)
            )
            self.action_space_n = len(env.action_space.nvec)

        if teammate_policy is None:
            # a function that returns an action to stay still no matter the input
            self.teammate_policy = lambda *_: 0
        else:
            self.teammate_policy = teammate_policy

        if opponent_policy is None:
            # a function that returns random actions no matter the input
            self.opponent_policy = lambda *_: self.env.action_space.sample()
        else:
            self.opponent_policy = opponent_policy

    def step(self, action):
        env_action = {
            # actions for team 2
            2: self.opponent_policy(self.last_obs[2]),
            3: self.opponent_policy(self.last_obs[3]),
        }
        if isinstance(self.action_space, gym.spaces.Discrete):
            if self.single_player:
                env_action[0] = action
                env_action[1] = self.teammate_policy(self.last_obs[1])
            else:
                env_action[0] = action // self.action_space_n
                env_action[1] = action % self.action_space_n
        else:
            env_action[0] = action[: self.action_space_n]
            env_action[1] = action[self.action_space_n :]

        obs, reward, done, info = self.env.step(env_action)

        return (
            self._preprocess_obs(obs),
            reward[0] + reward[1],
            done["__all__"],
            info[0],
        )

    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        self.last_obs = obs
        if self.single_player:
            return obs[0]
        else:
            return np.concatenate((obs[0], obs[1]))

    def set_opponent_policy(self, opponent_policy):
        self.opponent_policy = opponent_policy

    def set_teammate_policy(self, teammate_policy):
        self.teammate_policy = teammate_policy

    def set_policies(self, policy):
        self.set_opponent_policy(policy)
        self.set_teammate_policy(policy)


class ContinuousActionSpaceWrapper(gym.core.Wrapper):
    """
    A wrapper for continuous action space environment.
    Converts MultiDiscrete action space to Box action space.
    """

    def __init__(self, env):
        super(ContinuousActionSpaceWrapper, self).__init__(env)
        self.env = env
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.action_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(len(env.action_space.nvec),)
        )
        self.dims = [d - 1 for d in env.action_space.nvec]

    def step(self, action):
        if type(action) is dict:
            action = {i: self._preprocess_action(action[i]) for i in action}
        else:
            action = self._preprocess_action(action)
        return self.env.step(action)

    def _preprocess_action(self, action):
        # TODO implement function such that:
        """
        Translates continuous actions from range [-1, 1] (where -1 is reverse
        direction, 0 is no action, and 1 is forward direction) to (0, 1, 2)
        discrete values, where 0 is no action, 1 is forward, and 2 is reverse.
        """
        # Example:
        # return [int(round(v * d)) for v, d in zip(action, self.dims)]
        raise NotImplementedError(
            "ContinuousActionSpaceWrapper is not fully implemented yet"
        )


class EnvChannelWrapper(gym.core.Wrapper):
    def __init__(self, env, env_channel):
        super().__init__(env)
        self.env_channel = env_channel
