import logging
import os

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from soccer_twos.agent_interface import AgentInterface  # expose
from soccer_twos.package import check_package, TRAINING_ENV_PATH, ROLLOUT_ENV_PATH
from soccer_twos.wrappers import (
    MultiAgentUnityWrapper,
    MultiagentTeamWrapper,
    TeamVsPolicyWrapper,
    EnvType,
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
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(
        time_scale=env_config.get("time_scale", 20),  # 20x speedup
        quality_level=env_config.get("quality_level", 0),  # lowest
        target_frame_rate=-1,  # unbounded
        capture_frame_rate=60,  # 60 fps
    )

    unity_env = UnityEnvironment(
        env_config["env_path"],
        no_graphics=not env_config.get("render", False),
        base_port=env_config.get("base_port", 50039),
        worker_id=env_config.get("worker_id", 0),
        side_channels=[channel],
    )

    env = MultiAgentUnityWrapper(unity_env)

    if "variation" in env_config:
        if env_config["variation"] == EnvType.multiagent_player:
            return env
        elif env_config["variation"] == EnvType.multiagent_team:
            return MultiagentTeamWrapper(env)
        elif env_config["variation"] == EnvType.team_vs_policy:
            return TeamVsPolicyWrapper(
                env,
                env_config["opponent_policy"]
                if "opponent_policy" in env_config
                else None,
            )
        else:
            raise ValueError(
                "variation parameter invalid. Must be a EnvType member: ",
                [e.value for e in EnvType],
            )
    return env
