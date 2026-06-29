# start_selenium.py
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from logger_setup import setup_logger

# ================================================================================================
# Variables
# ================================================================================================

# XPath Selectors
XPATH_MENU_BUTTON = "/html/body/div[1]/div/main/div/div/div/main/div/div/div[1]/div/div[2]/div/button"
XPATH_NEW_CHAT_BUTTON = "/html/body/div[3]/div[4]/div[1]/button"
XPATH_START_NEW_CHAT_BUTTON = "/html/body/div[5]/div[3]/button"
XPATH_ENTER_MESSAGE_TEXTAREA = "/html/body/div[1]/div/main/div/div/div/main/div/div/div[2]/div/div[2]/div/div/div/div[1]/textarea"

# Timeout for waiting on elements to become clickable
DEFAULT_WAIT_TIME = 60

logger = setup_logger("selenium", "logs/selenium.log")

# ================================================================================================
# Helper Functions
# ================================================================================================

def wait_for_character_ai_page_load(driver, wait_time=DEFAULT_WAIT_TIME):
    """
    Wait until the Character.AI chat page is fully loaded by waiting for the presence
    of the chat input textarea. Logs a warning if the page doesn't load in time.
    """
    try:
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, XPATH_ENTER_MESSAGE_TEXTAREA))
        )
        logger.info("Character.AI chat page loaded successfully.")
    except Exception as e:
        logger.warning(f"Page didn't load in time: {e}")
        print(f"Page didn't load in time: {e}")

def setup_webdriver(debugger_address="localhost:9222", headless=False):
    """
    Set up and return a configured Chrome WebDriver instance.
    """
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", debugger_address)
    
    # Optional headless mode
    if headless:
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")

    # Chrome options
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def wait_for_element(driver, xpath, wait_time=DEFAULT_WAIT_TIME):
    """
    Wait for an element located by XPath to be visible and clickable.
    Return the element if found, otherwise None.
    """
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        return element
    except Exception as e:
        print(f"Element not interactable with XPath: {xpath}, Exception: {e}")
        logger.warning(f"Element not interactable with XPath: {xpath}, Exception: {e}")
        return None


def start_new_chat(driver):
    # Click the menu button
    menu_button = wait_for_element(driver, XPATH_MENU_BUTTON)
    if menu_button:
        menu_button.click()
        time.sleep(1.5)
    
        # Click the "New Chat" button
        new_chat_button = wait_for_element(driver, XPATH_NEW_CHAT_BUTTON)
        if new_chat_button:
            new_chat_button.click()
            time.sleep(1.5)

            # Click the "Start New Chat" button
            start_new_chat_button = wait_for_element(driver, XPATH_START_NEW_CHAT_BUTTON)
            if start_new_chat_button:
                start_new_chat_button.click()
                time.sleep(1.5)

                # Wait for chat input box to appear
                chat_input_xpath = XPATH_ENTER_MESSAGE_TEXTAREA 
                wait_for_element(driver, chat_input_xpath)

