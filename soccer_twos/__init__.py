import logging
import os

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from soccer_twos.agent_interface import AgentInterface  # expose
from soccer_twos.package import check_package, TRAINING_ENV_PATH, ROLLOUT_ENV_PATH
from soccer_twos.side_channels import EnvConfigurationChannel
from soccer_twos.utils import DummyEnv
from soccer_twos.wrappers import (
    MultiAgentUnityWrapper,
    MultiagentTeamWrapper,
    TeamVsPolicyWrapper,
    EnvType,
    TerminationMode,
    EnvChannelWrapper,
)

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

check_package()


def make(**env_config):
    """
    Creates a Unity environment with the given configuration.
    All args are optional.
    Args:
        render: Whether to render the environment. Defaults to False.
        watch: Whether to run an audience-friendly version the provided
            Soccer-Twos environment. Forces `render` to True, `time_scale` to 1 and
            `quality_level` to 5. Has no effect when `env_path` is set. Defaults to False.
        variation: A soccer env variation in EnvType. Defaults to `EnvType.multiagent_player`.
        time_scale: The time scale to use for the environment. This should be less
            than 100x for better simulation accuracy. Defaults to 20x realtime.
        quality_level: The quality level to use when rendering the environment.
            Ranges between 0 (lowest) and 5 (highest). Defaults to 0.
        env_path: The path to the environment executable. Overrides `watch`. Defaults
            to the provided Soccer-Twos environment.
        base_port: The base port to use to communicate with the environment. Defaults to 50039.
        worker_id: Used as base port shift to avoid communication conflicts. Defaults to 0.
        flatten_branched: If True, turn branched discrete action spaces into a Discrete space
            rather than MultiDiscrete. Defaults to False.
        opponent_policy: The policy to use for the opponent when `variation==team_vs_policy`.
            Defaults to a random agent.
        single_player: Whether to let the agent control a single player, while the other stays still.
            Only works when `variation==team_vs_policy`. Defaults to False.
        blue_team_name: The name of the blue team. Defaults to "BLUE".
        orange_team_name: The name of the orange team. Defaults to "ORANGE".
        env_channel: The side channel to use for communication with the environment. Defaults to None.
        uint8_visual: Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
        action_space_seed: If non-None, will be used to set the random seed on created gym.Space
            instances. Defaults to None.
        termination_mode: A string (enum) suggesting when to end an episode. Supports "ANY", "MAJORITY"
            and "ALL" which are atributes on `TerminationMode`. Defaults to TerminationMode.ANY.
    Returns: A multi-agent enabled, gym-friendly Unity environment.
    """

    # set watchable env if path is not provided
    if not env_config.get("env_path"):
        if env_config.get("watch"):
            env_config["env_path"] = ROLLOUT_ENV_PATH
            env_config["time_scale"] = 1
            env_config["quality_level"] = 5
            env_config["render"] = True
        else:
            env_config["env_path"] = TRAINING_ENV_PATH

    # set engine configs
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        time_scale=env_config.get("time_scale", 20),  # 20x speedup
        quality_level=env_config.get("quality_level", 0),  # lowest
    )
    # set env configs
    if env_config.get("env_channel"):
        env_channel = env_config.get("env_channel")
    else:
        env_channel = EnvConfigurationChannel()
    env_channel.set_parameters(
        blue_team_name=env_config.get("blue_team_name"),
        orange_team_name=env_config.get("orange_team_name"),
    )

    unity_env = UnityEnvironment(
        env_config["env_path"],
        no_graphics=not env_config.get("render", False),
        base_port=env_config.get("base_port", 50039),
        worker_id=env_config.get("worker_id", 0),
        side_channels=[engine_channel, env_channel],
    )

    multiagent_config = {
        k: env_config[k]
        for k in [
            "uint8_visual",
            "flatten_branched",
            "action_space_seed",
            "termination_mode",
        ]
        if k in env_config
    }
    env = MultiAgentUnityWrapper(unity_env, **multiagent_config)

    if "variation" in env_config:
        if EnvType(env_config["variation"]) is EnvType.multiagent_player:
            pass
        elif EnvType(env_config["variation"]) is EnvType.multiagent_team:
            env = MultiagentTeamWrapper(env)
        elif EnvType(env_config["variation"]) is EnvType.team_vs_policy:
            env = TeamVsPolicyWrapper(
                env,
                opponent_policy=env_config["opponent_policy"]
                if "opponent_policy" in env_config
                else None,
                single_player=env_config["single_player"]
                if "single_player" in env_config
                else False,
            )
        else:
            raise ValueError(
                "Variation parameter invalid. Must be an EnvType member: "
                + str([e.value for e in EnvType])
                + ". Received "
                + env_config["variation"]
            )

    env = EnvChannelWrapper(env, env_channel)
    return env
