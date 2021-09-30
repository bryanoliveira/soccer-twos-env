from typing import Any, Dict, List, Optional, Tuple, Union

from ray.rllib import MultiAgentEnv
import numpy as np
import gym
from gym import spaces
from gym_unity.envs import UnityToGymWrapper, UnityGymException, ActionFlattener
from mlagents_envs import logging_util
from mlagents_envs.base_env import ActionTuple, BaseEnv, DecisionSteps, TerminalSteps


logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.INFO)
GymStepResult = Tuple[np.ndarray, float, bool, Dict]


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
                # done and info uses team-wise info
                done = done or d
                info_dict[group_id] = i
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
        info = {"step": info}
        return observations, rewards, done, info

    def _set_action(self, action: List[Any], group_name: str) -> None:
        """Sets the action for an group within the environment.
        Args:
            action (list): the action to take
            group_name (str): the name of the group to set the action for
        """

        if self._flattener is not None:
            # Translate action into list
            action = self._flattener.lookup_action(action)

        action = np.array(action).reshape((-1, self.action_size))

        action_tuple = ActionTuple()
        if self.group_spec.action_spec.is_continuous():
            action_tuple.add_continuous(action)
        else:
            action_tuple.add_discrete(action)

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


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass
