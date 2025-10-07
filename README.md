# README.md
# Persona2Params

An OSS monolith MVP for simulating roleplaying dialogs between up to 5 LLM agents.

## Setup

1. Create a `.env` file with `OPENROUTER_API_KEY=your_key_here`.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Run: `python main.py` (or `python main.py --debug` for debug mode)

## Usage

- Enter context when prompted (e.g., "shakespeare's midsummer nights dream").
- Enter number of steps (max 25).
- Dialog proceeds, with optional debug output.
- Report saved as markdown file.

Uses OpenRouter API (OpenAI-compatible). Model hardcoded to Claude 3.5 Sonnet for quality.

## Notes

- Persona2Params integrated via Director's param adjustments.
- All realtime steering via LLM calls.
- Prompts in `prompts.json` for easy tweaking.