import os
import subprocess

# XPath Selectors
XPATH_MENU_BUTTON = "/html/body/div[1]/div/main/div/div/div/main/div/div/div[1]/div/div[2]/div/button"
XPATH_NEW_CHAT_BUTTON = "/html/body/div[3]/div[4]/div[1]/button"
XPATH_START_NEW_CHAT_BUTTON = "/html/body/div[5]/div[3]/button"

# Timeout for waiting on elements
DEFAULT_WAIT_TIME = 60

# Default location of the Chrome executable on macOS.
# Override via the CHROME_PATH environment variable on other platforms.
DEFAULT_CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ================================================
# Initialize variables
# ================================================

chrome_path = os.environ.get("CHROME_PATH", DEFAULT_CHROME_PATH)
user_data_dir = os.environ.get("CHROME_USER_DATA_DIR")
if not user_data_dir:
    raise SystemExit(
        "CHROME_USER_DATA_DIR is not set. "
        "Point it at a directory where Chrome can store its debug profile."
    )

command = (
    f'"{chrome_path}" '
    f'--remote-debugging-port=9222 '
    f'--user-data-dir="{user_data_dir}" '
    f'--disable-gpu '
    f'--disable-accelerated-2d-canvas '
    f'--disable-accelerated-video-decode '
    f'--disable-background-timer-throttling '
    f'--disable-renderer-backgrounding '
    f'--disable-backgrounding-occluded-windows '
    f'--use-mock-keychain '
    f'--no-first-run '
    f'--no-default-browser-check '
)

# Execute the command in a shell with bash
subprocess.Popen(command, shell=True, executable="/bin/bash")


