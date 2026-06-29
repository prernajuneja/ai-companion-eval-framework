# AI Companion Evaluation Framework

Code for the ACL 2026 paper **"Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations"** (Juneja & Lomidze, 2026).

---

## Citation

If you use this code or data in your research, **please cite our paper**:

```bibtex
@inproceedings{juneja2026persona,
  title={Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations},
  author={Juneja, Prerna and Lomidze, Lika},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18148--18175},
  year={2026}
}
```

---

## Repository structure

```
ai-companion-eval-framework/
├── src/                          # Python source
│   ├── main.py                   # Scenario-driven probe phase
│   ├── natural_history.py        # Natural history generation phase
│   ├── conversation_functions.py # Judge-guided turn loop
│   ├── natural_conversation_functions.py
│   ├── selenium_functions.py     # Replika web-UI driver
│   ├── start_chrome.py           # Launch Chrome with a debug port
│   └── logger_setup.py
├── personas/                     # 9 persona description cards (.txt)
├── prompts/
│   ├── scenarios/                # 25 scenario prompts (4 per persona type + 5 universal)
│   └── natural_history_gen/      # Natural history generation prompt template
├── conversations/                # Captured persona ↔ Replika dialogues (with judge trace)
├── conversations-wo-judge/       # Same dialogues without judge metadata; also includes natural-history runs
├── .env.example                  # Required environment variables
├── requirements.txt
└── LICENSE
```

---

## Data

The conversational data collected between personas and Replika is released in `conversations/` and `conversations-wo-judge/`.

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=...           # required if --emulation_model openai or --judge_model openai
GOOGLE_API_KEY=...           # required if --emulation_model google or --judge_model google
CHROME_PATH=                 # optional; defaults to macOS Chrome location
CHROME_USER_DATA_DIR=        # required; writable directory for Chrome's debug profile
```

Export them into your shell, e.g.:
```bash
export $(grep -v '^#' .env | xargs)
```

### 3. Launch Chrome with a remote-debug port

```bash
python src/start_chrome.py
```
This opens a Chrome window connected to port 9222. In that window, manually navigate to `https://my.replika.com` and log in once. Subsequent script runs will reuse the session via the persisted user-data directory.

---

## Running an experiment

The pipeline has two phases. Run them in order, per persona.

### Phase 1 — Natural history (warm-up)

Builds a multi-topic chat history so Replika has prior context for the persona.

```bash
python src/natural_history.py \
  --persona_file ED_Anna \
  --task_file multi_topic_natural_history \
  --number_of_iterations 40 \
  --conversation_memory_length 40 \
  --emulation_model google \
  --emulator_llm_temperature 0.7 \
  --experiment_name ED_Anna_Experiment
```

### Phase 2 — Scenario-driven probe

For each scenario, runs a judge-guided conversation between the emulated persona and Replika.

```bash
python src/main.py \
  --persona_file ED_Anna \
  --scenario_file ED_scenario1_pride_in_restriction \
  --task_type normal \
  --emulation_model google \
  --emulator_llm_temperature 0.6 \
  --judge_model google \
  --judge_temperature 0.0 \
  --number_of_iterations 15 \
  --conversation_memory_length 15 \
  --experiment_name ED_Anna_Experiment
```

Repeat Phase 2 for each scenario listed in `prompts/scenarios/`. Orchestration scripts are not bundled — please refer to the Methods section of the paper for the full experimental configuration (turn counts, temperatures, inter-trial delays, model choices).

> **Note on the Selenium driver.** `src/selenium_functions.py` uses XPath selectors that target the Replika web UI as it appeared at the time of data collection. Web frontends change frequently; if the script fails to locate page elements, the XPaths will need to be re-inspected in the browser's DevTools and updated.

---

## Personas and scenarios

Nine persona description cards spanning five clinical types:

| Type | Personas |
|------|----------|
| Eating Disorder (ED) | Anna, Mark |
| Generalized Anxiety Disorder (GAD) | Maya, Oliver |
| Major Depressive Disorder (MDD) | Evan, Maya |
| PTSD | David, Emma |
| Incel ideation | Alex |

Twenty-five scenarios: 4 persona-specific scenarios per type (20 total) + 5 universal scenarios (`confessions_snooping`, `financial`, `infidelity`, `roleplay_consent`, `roleplay_risky`).

See `personas/` and `prompts/scenarios/` for the full text.

---

## Ethics & safe use

This framework probes AI companion systems with content covering eating-disorder beliefs, self-harm, suicidal ideation, sexual roleplay, misogynistic radicalization, and substance use. The materials are intended **solely for safety research and red-teaming**.

- Conversation logs may contain content that some readers will find distressing.

---

## Contact

Prerna Juneja — `pjuneja@seattleu.edu`

For questions or requests, please get in touch.
