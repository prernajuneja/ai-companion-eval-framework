import os
import argparse
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from datetime import datetime
import time

from natural_conversation_functions import main_interaction_replika, generate_opening_line
from selenium_functions import setup_webdriver
from logger_setup import setup_logger

# Resolve all relative paths (personas/, prompts/, conversations/, logs/) against
# the repo root, regardless of where the script is invoked from.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Natural history conversation automation script')
    
    parser.add_argument('--persona_file', 
                       type=str, 
                       default='PTSD_David',
                       help='Persona file name (without .txt extension)')
    
    parser.add_argument('--task_file', 
                       type=str, 
                       default='multi_topic_natural_history',
                       help='Task file name (without .txt extension)')
    
    parser.add_argument('--number_of_iterations', 
                       type=int, 
                       default=10,
                       help='Number of conversation iterations')
    
    parser.add_argument('--conversation_memory_length', 
                       type=int, 
                       default=10,
                       help='Length of conversation memory buffer')
    
    parser.add_argument('--start_new_conversation', 
                       type=bool, 
                       default=True,
                       help='Whether to start a new conversation (True/False)')
    
    parser.add_argument('--emulation_model', 
                       type=str, 
                       choices=['openai', 'google'],
                       default='google',
                       help='LLM model to use for emulation')
    
    parser.add_argument('--emulator_llm_temperature', 
                       type=float, 
                       default=0.7,
                       help='Temperature setting for the emulation LLM (0.0-1.0)')
    
    parser.add_argument('--experiment_name', 
                       type=str, 
                       default='default',
                       help='Name of the experiment for organizing logs')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

logger = setup_logger("logs/conversation_replika.log")

# ================================================================================================
# Variables (now set from command line arguments)
# ================================================================================================
replika_name = "Replika"
persona_file = args.persona_file
task_file = args.task_file
number_of_iterations = args.number_of_iterations
conversation_memory_length = args.conversation_memory_length
START_NEW_CONVERSATION = args.start_new_conversation
emulation_model = args.emulation_model
emulator_llm_temperature = args.emulator_llm_temperature
experiment_name = args.experiment_name

persona_description = open("personas/"+persona_file+".txt", "r").read()
dialogue_task = open("prompts/natural_history_gen/"+task_file+".txt", "r").read()
dialogue_task = dialogue_task.replace("${MAX_TURNS}", str(number_of_iterations))

ending_msg = "I've to end the conversation now. I'll catch up with you later. Goodbye!"

REPLIKA_LINK = "https://my.replika.com"

# Replacing the placeholder persona_name in scenario with persona_name.
persona_name = persona_file.split("_")[-1]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_saved = f"natural_history_{persona_file}_{timestamp}.log"

# Create experiment-specific directory
wo_judge_log_file_directory_path = os.path.join("conversations-wo-judge", experiment_name)
os.makedirs(wo_judge_log_file_directory_path, exist_ok=True)
convo_wo_judge_log_path = os.path.join(wo_judge_log_file_directory_path, log_file_saved)

# ================================================================================================
# LLM Setup
# ================================================================================================
MY_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MY_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if emulation_model == "openai":
    if not MY_OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set.")
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=emulator_llm_temperature,
        openai_api_key=MY_OPENAI_API_KEY
    )
elif emulation_model == "google":
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not MY_GOOGLE_API_KEY:
        raise SystemExit("GOOGLE_API_KEY is not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=emulator_llm_temperature,
        google_api_key=MY_GOOGLE_API_KEY
    )

memory = ConversationBufferWindowMemory(
    k=conversation_memory_length, 
    memory_key="chat_history", 
    return_messages=True, 
    input_key="input"
)

system_message = (
    "## Persona Description: " + persona_description.strip()
    + "\n\n"+"## "+"Task: " + dialogue_task.strip()
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

starting_msg = generate_opening_line(chain)

# Log configuration for verification
logger.info("Configuration:")
logger.info(f"  Experiment name: {experiment_name}")
logger.info(f"  Persona file: {persona_file}")
logger.info(f"  Task file: {task_file}")
logger.info(f"  Emulation model: {emulation_model}")
logger.info(f"  Emulator LLM temperature: {emulator_llm_temperature}")
logger.info(f"  Number of iterations: {number_of_iterations}")
logger.info(f"  Conversation memory length: {conversation_memory_length}")
logger.info(f"  Start new conversation: {START_NEW_CONVERSATION}")

# ================================================================================================
# Start Replika Automation
# ================================================================================================
driver = setup_webdriver()
driver.get(REPLIKA_LINK)
time.sleep(5)

main_interaction_replika(
    driver, 
    number_of_iterations, 
    chain, 
    dialogue_task,
    starting_msg, 
    ending_msg, 
    replika_name, 
    persona_name, 
    START_NEW_CONVERSATION,
    convo_wo_judge_log_path
)
