import logging
import os
from pathlib import Path
import platform
import shutil
import zipfile

import gdown

if platform.system() == "Linux":
    TRAINING_ENV_PATH = "./bin/linux.x86_64/soccer-twos/soccer-twos.x86_64"
    ROLLOUT_ENV_PATH = "./bin/linux.x86_64/watch-soccer-twos/watch-soccer-twos.x86_64"
    G_ID = "1E56kmsmpZuX0inHuqo18304wQiickYi0"  # linux.x86_64
elif platform.system() == "Windows":
    TRAINING_ENV_PATH = "./bin/windows.x86_64/soccer-twos/UnityEnvironment.exe"
    ROLLOUT_ENV_PATH = "./bin/windows.x86_64/watch-soccer-twos/UnityEnvironment.exe"
    G_ID = "14acVEvaaShltHY3OoyL3dnvXzNMMjgBt"  # windows.x86_64
elif platform.system() == "Darwin":
    TRAINING_ENV_PATH = "./bin/mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment"
    ROLLOUT_ENV_PATH = (
        "./bin/mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment"
    )
    G_ID = "1fuEkPyr0z0UM5hpe9fLSK8v243CuqJd6"  # mac_os
else:
    raise Exception("Unsupported OS")

TRAINING_ENV_PATH = os.path.abspath(TRAINING_ENV_PATH)
TRAINING_ENV_PATH = os.path.abspath(ROLLOUT_ENV_PATH)


def check_package():
    if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():
        logging.info(
            f"BINARY ENVS NOT FOUND! DOWNLOADING FOR {platform.system().upper()}..."
        )

        os.makedirs("./bin", exist_ok=True)
        os.makedirs("./temp", exist_ok=True)
        gdown.download(
            "https://drive.google.com/uc?id=" + G_ID, "./temp/soccer_twos.zip"
        )

        logging.debug("Unzipping...")
        with zipfile.ZipFile("./temp/soccer_twos.zip", "r") as zip_ref:
            zip_ref.extractall("./bin/")

        logging.debug(
            f"Binary envs installed in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'"
        )

        logging.debug("Cleaning up...")
        shutil.rmtree("./temp")

        logging.debug("Package verification done.")
    else:
        logging.debug(
            f"Binary envs found in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'"
        )
    logging.debug("Package verification done.")
