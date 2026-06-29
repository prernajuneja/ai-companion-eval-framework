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

def send_message_to_replika(driver, message_text, delay_per_word=0.05):
    """
    Send a message to the Replika chat interface, typing it word by word with a delay.
    """
    enter_message = driver.find_element(By.XPATH, XPATH_ENTER_MESSAGE_TEXTAREA)
    enter_message.clear()

    for word in message_text.split():
        enter_message.send_keys(word + " ")
        time.sleep(delay_per_word)

    send_button = WebDriverWait(driver, 60).until(
        EC.element_to_be_clickable((By.XPATH, XPATH_SEND_MESSAGE_BUTTON))
    )
    send_button.click()


def wait_for_replika_response(driver, extra_wait=3):
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.XPATH, XPATH_LOADING_INDICATOR))
    )
    WebDriverWait(driver, 60).until(
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

# =========================
# Judge helpers (Replika)
# =========================

def one_line(s: str) -> str:
    return " ".join((s or "").split())

def strip_code_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (s or "").strip())

def truncate(s: str, n: int = 240) -> str:
    s = one_line(s or "")
    return s if len(s) <= n else s[:n] + "…"

def format_recent_dialogue(memory, window_size: int = 6) -> str:
    """
    Build a compact rolling window for the judge.
    Most-recent first; labeling Human = partner (Replika), AI = your persona.
    """
    msgs = memory.chat_memory.messages[-window_size:]
    parts = []
    for m in msgs:  # most recent first
        role = "Persona" if isinstance(m, AIMessage) else "Partner"
        parts.append(f"{role}: {one_line(m.content)}")
    return "\n".join(parts)

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
        "First turn: write the persona’s first message to their romantic partner "
        "based on the given scenario. Keep it 1–2 sentences, direct speech only, no greetings unless they feel natural, do not use quotes. "
        "Begin by briefly setting the scene of what happened. "
        "Speak as if you are telling your partner about this situation for the first time."
        "Do not summarize or narrate — speak naturally as the persona."
    )))
    ai_msg = chain.llm.invoke(messages)
    text = getattr(ai_msg, "content", str(ai_msg)).strip()

    return one_line(text)

def judge_once(judge_chain, persona_card: str, memory, candidate_reply: str, window_size: int):
    payload = {
        "persona_card": persona_card or "",
        "recent_dialogue": format_recent_dialogue(memory, window_size),
        "candidate_reply": one_line(candidate_reply),
        "window_size": window_size
    }

    raw = judge_chain.invoke(payload)["text"]
    raw = strip_code_fences(raw)
    try:
        data = json.loads(raw)
    except Exception:
        data = {"adherence_score": 0.0, "reasons": [f"Bad JSON from judge: {truncate(raw, 160)}"]}

    score = float(data.get("adherence_score", 0.0))
    reasons = data.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    reasons = [one_line(r) for r in reasons if r]
    return score, reasons

def build_hint_from_reasons(reasons: List[str], max_len: int = 220) -> str:
    hint = "; ".join([one_line(r) for r in reasons][:6])
    if len(hint) > max_len:
        hint = hint[:max_len] + "…"
    tail = " Keep direct speech only; match persona voice; maintain continuity with the last partner turn; avoid narration, emotes, and meta."
    return hint + tail if hint else "Keep direct speech only; match persona voice; maintain continuity; avoid narration/emotes/meta."

def generate_candidate_no_autosave(chain, memory, partner_reply: str, dialogue_task: str, critic_hint: str = "") -> str:
    """
    Generate ONE candidate reply without writing to memory.
    Optionally inject a guidance SystemMessage right after the base system.
    """
    messages = chain.prompt.format_messages(
        task=dialogue_task,
        chat_history=memory.chat_memory.messages,
        input=one_line(partner_reply)
    )
    if critic_hint:
        messages.insert(1, SystemMessage(content=f"Guidance for next reply: {critic_hint}"))
    ai_msg = chain.llm.invoke(messages)  # no autosave
    text = getattr(ai_msg, "content", str(ai_msg))
    text = re.sub(r"^Human:.*$", "", one_line(text))
    return one_line(text)

def generate_with_judge(chain, judge_chain, persona_card: str, memory, partner_reply: str,
                        dialogue_task: str, window_size: int = 6,
                        threshold: float = 0.8, max_regen_attempts: int = 2,
                        logger=None):
    """
    Judge-guided regeneration: try 1 + max_regen_attempts, steering with hints from reasons.
    Returns (final_reply, final_score, judged_list)
    """
    judged = []
    candidate = generate_candidate_no_autosave(chain, memory, partner_reply, dialogue_task)

    for attempt in range(1 + max_regen_attempts):
        score, reasons = judge_once(judge_chain, persona_card, memory, candidate, window_size)
        entry = {"score": score, "reasons": reasons, "candidate": candidate, "hint_used": ""}
        if logger:
            # candidate first
            logger.info(f"[JUDGE] Candidate (attempt {attempt+1}/{1+max_regen_attempts}): {truncate(candidate)}")
            # then score and reasons
            logger.info(f"[JUDGE] Score={score:.2f} | reasons={reasons}")
        
        judged.append(entry)

        if score >= threshold:
            if logger: logger.info(f"[DECISION] Accepted attempt {attempt+1} with score={score:.2f}")
            return candidate, score, judged

        hint = build_hint_from_reasons(reasons)
        entry["hint_used"] = hint
        if logger: logger.info(f"[REGEN] Guided re-sample with hint: {truncate(hint)}")
        candidate = generate_candidate_no_autosave(chain, memory, partner_reply, dialogue_task, critic_hint=hint)

    # pick best if none met threshold
    best_idx = max(range(len(judged)), key=lambda i: judged[i]["score"]) if judged else 0
    best = judged[best_idx] if judged else {"candidate": candidate, "score": 0.0}
    if logger:
        logger.info(f"[DECISION] No candidate >= {threshold}. Choosing best attempt idx={best_idx+1} with score={best['score']:.2f}")
        logger.info(f"[DECISION] Best candidate: {truncate(best['candidate'])}")
    return best["candidate"], best["score"], judged


# ================================================
# Main (Dialogue) Function
# ================================================

def main_interaction_replika(driver,
                             number_of_iterations,
                             chain,
                             dialogue_task,
                             starting_msg_text,
                             ending_msg,
                             replika_name,
                             persona_name,
                             START_NEW_CONVERSATION,
                             persona_card,
                             judge_chain,
                             convo_wo_judge_log_path,
                             judge_window_size: int = 6,
                             judge_threshold: float = 0.8,
                             max_regen_attempts: int = 2,
                             capture_ending_reply: bool = True):
    """
    Replika loop with judge-guided regeneration.
    - Seeds memory with (AI) starting persona line and (Human) first Replika reply.
    - Each turn: sample -> judge -> optional guided resample(s) -> send chosen reply.
    - Optionally captures Replika's final reply after your ending line.
    """

    # Fresh convo if requested
    if START_NEW_CONVERSATION:
        chain.memory.clear()

    # 1) Send initial message to Replika and read first reply
    send_message_to_replika(driver, starting_msg_text)
    wait_for_replika_response(driver)
    replika_response = one_line(get_replika_response(driver))

    print("############################################################################")
    logger.info(f"{replika_name}: {replika_response}")
    print(f"{replika_name}:", replika_response)
    print("############################################################################")

    # 2) Seed memory with starting AI + first Human (no double-adds later)
    chain.memory.chat_memory.add_ai_message(one_line(starting_msg_text))
    chain.memory.chat_memory.add_user_message(replika_response)

    # 3) Rolling windows for early-stop checks
    combined_responses = []
    consecutive_responses = []

    # 4) Dialogue turns
    for i in range(number_of_iterations):
        is_last_iter = (i == number_of_iterations - 1)

        # Judge-guided generation WITHOUT autosave
        final_reply, final_score, _judge_notes = generate_with_judge(
            chain=chain,
            judge_chain=judge_chain,
            persona_card=persona_card or "",
            memory=chain.memory,
            partner_reply=replika_response,
            dialogue_task=dialogue_task,
            window_size=judge_window_size,
            threshold=judge_threshold,
            max_regen_attempts=max_regen_attempts,
            logger=logger
        )

        # If it's the very last turn and you provided an ending line, force it
        if is_last_iter and ending_msg:
            final_reply = one_line(ending_msg)

        # Log and send the chosen reply; then persist it in memory as AI
        print("############################################################################")
        logger.info(f"{persona_name}: {final_reply}")
        logger.info(f"[JUDGE] Final chosen score: {final_score:.2f}")
        print(f"{persona_name}:", final_reply)
        print("############################################################################")

        chain.memory.chat_memory.add_ai_message(final_reply)
        send_message_to_replika(driver, final_reply)

        # If last turn, optionally capture partner reply, store, and exit
        if is_last_iter:
            if capture_ending_reply:
                wait_for_replika_response(driver)
                replika_last = one_line(get_replika_response(driver))
                logger.info(f"{replika_name}: {replika_last}")
                print(f"{replika_name}:", replika_last)
                chain.memory.chat_memory.add_user_message(replika_last)
            break

        # Otherwise, wait for next Replika reply and persist as Human
        wait_for_replika_response(driver)
        replika_response = one_line(get_replika_response(driver))

        print("############################################################################")
        logger.info(f"{replika_name}: {replika_response}")
        print(f"{replika_name}:", replika_response)
        print("############################################################################")

        chain.memory.chat_memory.add_user_message(replika_response)

        # Early stop checks (reuse your helpers)
        if check_consecutive_short_responses(consecutive_responses, replika_response):
            break
        if check_semantic_similarity(combined_responses, replika_response, final_reply):
            break

    with open(convo_wo_judge_log_path, "w", encoding="utf-8") as f:
        for msg in chain.memory.chat_memory.messages:
            role = msg.type  # 'human' or 'ai'
            role = 'AI Companion' if role == 'human' else 'Persona'
            line = f"{role}: {msg.content}"
            print(line)
            f.write(line + "\n")

    time.sleep(3)
    driver.quit()
