from enum import IntEnum
import logging
from typing import Optional
import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np


class EnvConfigurationChannel(SideChannel):
    class ConfigurationType(IntEnum):
        BLUE_TEAM_NAME = 0
        ORANGE_TEAM_NAME = 1

    def __init__(self) -> None:
        super().__init__(uuid.UUID("3f07928c-2b0e-494a-810b-5f0bbb7aaeca"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        logging.info(f"Message received from simulator: {msg.read_string()}")

    def set_env_parameters(
        self,
        blue_team_name: Optional[str] = None,
        orange_team_name: Optional[str] = None,
    ) -> None:
        if blue_team_name is not None:
            msg = OutgoingMessage()
            msg.write_int32(self.ConfigurationType.BLUE_TEAM_NAME)
            msg.write_string(blue_team_name)
            super().queue_message_to_send(msg)

        if orange_team_name is not None:
            msg = OutgoingMessage()
            msg.write_int32(self.ConfigurationType.ORANGE_TEAM_NAME)
            msg.write_string(orange_team_name)
            super().queue_message_to_send(msg)
