import logging
import os
from pathlib import Path
import platform
import shutil
import stat
import zipfile
import requests, zipfile, io

if platform.system() == "Linux":
    TRAINING_ENV_PATH = "linux-x86_64/soccer-twos/soccer-twos.x86_64"
    ROLLOUT_ENV_PATH = "linux-x86_64/watch-soccer-twos/watch-soccer-twos.x86_64"
    ZIP_URL = "https://github.com/bryanoliveira/soccer-twos-env/releases/download/0.1.13/linux-x86_64.zip"  # linux.x86_64
elif platform.system() == "Windows":
    TRAINING_ENV_PATH = "windows-x86_64/soccer-twos/UnityEnvironment.exe"
    ROLLOUT_ENV_PATH = "windows-x86_64/watch-soccer-twos/UnityEnvironment.exe"
    ZIP_URL = "https://github.com/bryanoliveira/soccer-twos-env/releases/download/0.1.13/windows-x86_64.zip"  # windows.x86_64
elif platform.system() == "Darwin":
    TRAINING_ENV_PATH = "mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment"
    ROLLOUT_ENV_PATH = "mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment"
    ZIP_URL = "https://github.com/bryanoliveira/soccer-twos-env/releases/download/0.1.13/mac_os.zip"  # mac_os
else:
    raise Exception("Unsupported OS")

__ENV_VERSION = "v2"
__CURR_DIR = os.path.dirname(os.path.abspath(__file__))
__BIN_DIR = os.path.join(__CURR_DIR, "bin", __ENV_VERSION)
TRAINING_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, TRAINING_ENV_PATH))
ROLLOUT_ENV_PATH = os.path.abspath(os.path.join(__BIN_DIR, ROLLOUT_ENV_PATH))


def check_package():
    """
    Checks if the package is installed and, if not, installs it according to the
    current platform.
    """

    if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():
        logging.info(
            f"BINARY ENVS NOT FOUND! DOWNLOADING FOR {platform.system().upper()}..."
        )

        os.makedirs(__BIN_DIR, exist_ok=True)
        os.makedirs(os.path.join(__CURR_DIR, "temp"), exist_ok=True)

        req = requests.get(ZIP_URL)
        zip_ref = zipfile.ZipFile(io.BytesIO(req.content))
        logging.debug("Unzipping...")
        zip_ref.extractall(__BIN_DIR)            

        try:
            # try adding execute permissions to the binary
            st = os.stat(TRAINING_ENV_PATH)
            os.chmod(TRAINING_ENV_PATH, st.st_mode | stat.S_IEXEC)
            st = os.stat(ROLLOUT_ENV_PATH)
            os.chmod(ROLLOUT_ENV_PATH, st.st_mode | stat.S_IEXEC)
        except Exception as e:
            logging.warn(e)
            pass

        logging.debug(
            f"Binary envs installed in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'"
        )

        logging.debug("Cleaning up...")
        shutil.rmtree(os.path.join(__CURR_DIR, "./temp"))

        logging.debug("Package verification done.")
        logging.info(
            f"Soccer environment installed for {platform.system().upper()}."
        )
    else:
        logging.debug(
            f"Binary envs found in '{TRAINING_ENV_PATH}' and '{ROLLOUT_ENV_PATH}'"
        )
    logging.debug("Package verification done.")
