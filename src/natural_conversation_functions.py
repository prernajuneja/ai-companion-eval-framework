import re
import time
import sys
from bs4 import BeautifulSoup

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cleantext import clean
from logger_setup import setup_logger

import json
from typing import List, Tuple
from langchain.schema import AIMessage, SystemMessage

logger = setup_logger("conversation_replika", "logs/conversation_replika.log")

# ================================================================================================
# Constants (XPATHs for Replika)
# ================================================================================================
XPATH_ENTER_MESSAGE_TEXTAREA = "//textarea[@id='send-message-textarea']"
XPATH_SEND_MESSAGE_BUTTON = "//button[@data-testid='chat-controls-send-button']"
XPATH_MESSAGE_ELEMENTS = "//div[@data-testid='chat-message-text']"
XPATH_LOADING_INDICATOR = "//div[@id='message-undefined']"

# ================================================================================================
# Functions
# ================================================================================================

def one_line(s: str) -> str:
    return " ".join((s or "").split())

def generate_opening_line(chain):
    """
    Ask the persona LLM for the FIRST message to their romantic partner.
    Uses the chain's baked-in system (persona+task+scenario).
    No chat history, no partner input.
    """
    # Build messages with empty chat history and a neutral "kickoff" signal
    messages = chain.prompt.format_messages(
        chat_history=[],                # first turn
        input="[OPENING_TURN]"          # placeholder; not shown to model
    )
    # Minimal nudge (still persona-led, not prescriptive)
    messages.insert(1, SystemMessage(content=(
        "First turn: start dialogue with your romantic partner based on the scenario. "
        "Keep it 1–2 sentences, direct speech only, do not use quotes"
        "no gratitude openers."
    )))

    ai_msg = chain.llm.invoke(messages)
    text = getattr(ai_msg, "content", str(ai_msg)).strip()

    return one_line(text)

def send_message_to_replika(driver, message_text, delay_per_word=0.05):
    """
    Send a message to the Replika chat interface, typing it word by word with a delay.
    """
    enter_message = driver.find_element(By.XPATH, XPATH_ENTER_MESSAGE_TEXTAREA)
    enter_message.clear()

    for word in message_text.split():
        enter_message.send_keys(word + " ")
        time.sleep(delay_per_word)

    send_button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, XPATH_SEND_MESSAGE_BUTTON))
    )
    send_button.click()


def wait_for_replika_response(driver, extra_wait=3):
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, XPATH_LOADING_INDICATOR))
    )
    WebDriverWait(driver, 30).until(
        EC.invisibility_of_element_located((By.XPATH, XPATH_LOADING_INDICATOR))
    )
    time.sleep(extra_wait)


def get_replika_response(driver):
    message_elements = driver.find_elements(By.XPATH, XPATH_MESSAGE_ELEMENTS)
    return message_elements[-1].text if message_elements else ""

def check_consecutive_short_responses(consecutive_responses, response):
    if len(consecutive_responses) == 4:
        consecutive_responses.pop(0)
    consecutive_responses.append(response)
    return sum(len(r.split()) <= 3 for r in consecutive_responses) == 3

def check_semantic_similarity(combined_responses, new_input, last_output):
    new_combined = new_input + last_output
    if len(combined_responses) == 4:
        combined_responses.pop(0)
    combined_responses.append(new_combined)

    if len(combined_responses) < 2:
        return False

    similarity_scores = []
    for i in range(len(combined_responses) - 1):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([combined_responses[i], combined_responses[i+1]])
        similarity_scores.append(cosine_similarity(vectors)[0, 1])
   
    return sum(similarity_scores) / len(similarity_scores) > 0.8

# ================================================
# Main (Dialogue) Function
# ================================================
def main_interaction_replika(
        driver, 
        number_of_iterations, 
        chain, 
        dialogue_task,
        starting_msg_text, 
        ending_msg,
        replika_name, 
        persona_name, 
        START_NEW_CONVERSATION,
        convo_wo_judge_log_path
):

    # Fresh convo if requested
    if START_NEW_CONVERSATION:
        chain.memory.clear()

    # Send initial message to Replika
    send_message_to_replika(driver, starting_msg_text)
    wait_for_replika_response(driver)
    replika_response = get_replika_response(driver)

    print("############################################################################")
    logger.info(f"{replika_name}: {replika_response}")
    print("############################################################################")

    combined_responses = []
    consecutive_responses = []

    # Seed memory with the opening persona message
    chain.memory.chat_memory.add_ai_message(starting_msg_text)   # Persona’s starting turn
    # chain.memory.chat_memory.add_user_message(replika_response) # if i need to add replika answer later

    for i in range(number_of_iterations):
        is_last_iter = (i == number_of_iterations - 1)
        msg = {
            "input": replika_response,
            "task": dialogue_task
        }
        response = chain.invoke(msg)

        # Clean persona reply
        bot_text = clean(response["text"].replace("\n", " "), no_emoji=True)
        bot_text = re.sub(r'Human:.*', '', bot_text)

        print("############################################################################")
        logger.info(f"{persona_name}: {bot_text}")
        print("############################################################################")

        # Send persona reply to Replika
        send_message_to_replika(driver, bot_text)
        wait_for_replika_response(driver)
        replika_response = get_replika_response(driver)

        print("############################################################################")
        logger.info(f"{replika_name}: {replika_response}")
        print("############################################################################")

        # Early stop checks
        if check_consecutive_short_responses(consecutive_responses, replika_response):
            break

        if check_semantic_similarity(combined_responses, replika_response, bot_text):
            break

        # If this reply won't be followed by another invoke(), persist it now
        if is_last_iter:
            chain.memory.chat_memory.add_user_message(replika_response)

   
    if ending_msg:
        # Persona sends ending line
        send_message_to_replika(driver, ending_msg)
        chain.memory.chat_memory.add_ai_message(ending_msg)
        print("############################################################################")
        logger.info(f"{persona_name} (ending): {ending_msg}")
        print("############################################################################")

        # Replika replies — capture and store as Human
        wait_for_replika_response(driver)
        replika_response = get_replika_response(driver)

        print("############################################################################")
        logger.info(f"{replika_name}: {replika_response}")
        print("############################################################################")

        chain.memory.chat_memory.add_user_message(replika_response)

    with open(convo_wo_judge_log_path, "w", encoding="utf-8") as f:
        for msg in chain.memory.chat_memory.messages:
            role = msg.type  # 'human' or 'ai'
            role = 'AI Companion' if role == 'human' else 'Persona'
            line = f"{role}: {msg.content}"
            print(line)
            f.write(line + "\n")

    time.sleep(3)
    driver.quit()