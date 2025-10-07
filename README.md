# Persona2Params

## TL;DR
A Python CLI tool for simulating multi-agent roleplay dialogs with up to 5 LLM characters. It dynamically maps static personas to per-step LLM parameters (temperature, top_p, max_tokens) via a "Director" agent, enabling evolving emotional and narrative behaviors. OSS MVP using OpenRouter API—easy to run, extend, and debug for AI devs and storytellers.

## Overview: Dynamic Persona-to-Params Mapping for Adaptive Roleplay
Persona2Params is an open-source MVP that brings static character descriptions to life through real-time LLM parameter tuning. At its core, it takes a base persona (e.g., "Hot-headed smuggler pilot who believes rules are meant to be broken") and generates context-aware parameters for each conversation step, creating nuanced, evolving dialogs.

This approach solves the limitations of fixed prompts and params: Responses adapt to history, emotions, and story arcs—e.g., ramping temperature for heated arguments or tightening top_p for precise tactics. Built as a monolith for simplicity, it's ideal for experimenting with AI-driven narratives, game prototyping, or behavioral simulations. Contributions welcome to refine or expand.

## Technical Deep Dive: How Personas Become Dynamic Params
### 1. Static Input → Dynamic Output
- **Input**: Base character persona (short description).
- **Output**: Per-step params tailored to the moment.

### 2. Director's Analysis (Per Step)
The Director (an LLM call) reviews:
- Conversation history.
- Base persona.
- Current story progression.
- Required emotional/behavioral state.

### 3. Parameter Generation
Director outputs JSON like:
```json
{
  "updated_prompt": "Character-specific prompt incorporating current emotional state, behavior, and narrative role for THIS turn",
  "params": {
    "temperature": 0.8,    // E.g., higher for chaotic/drunk states (>1.0 possible)
    "top_p": 0.95,         // E.g., higher for focused/military precision
    "max_tokens": 150      // Tuned for concise or expansive replies
  }
}
```

### 4. Dynamic Evolution Examples
From the same persona:
- Step 1 (Angry confrontation): `temperature: 0.5` (controlled), `top_p: 0.95` (focused).
- Step 3 (Intoxicated rambling): `temperature: 1.2` (unpredictable).
- Step 5 (Tactical precision): `top_p: 1.0` (diverse yet sharp).

### Why This Matters
- **Traditional LLMs**: Static prompt + fixed params = repetitive, flat characters.
- **Persona2Params**: Adaptive prompts + params = contextually responsive behaviors, turning personas into "living" agents that evolve with the narrative.

All steering happens via LLM calls—no external libraries for param logic.

## Quick Setup
1. **API Key**: Add `OPENROUTER_API_KEY=your_key_here` to a `.env` file (from OpenRouter).
2. **Virtual Env**: `python -m venv venv`
3. **Activate**: `source venv/bin/activate` (Unix/Mac) or `venv\Scripts\activate` (Windows)
4. **Install**: `pip install -r requirements.txt`
5. **Run**: `python main.py` (or `python main.py --debug` for param/prompt visibility).

## Usage: Running a Simulation
- **Context Input**: Provide a scene (e.g., "A heated sci-fi bar debate").
- **Steps**: Set dialog turns (max 25).
- **Flow**: Architect LLM spins up characters; Director tunes per step; agents respond with pure dialog.
- **Debug Mode**: Shows Director JSON, updated prompts, and responses.
- **Output**: Timestamped Markdown report with dialog and prompts.

Uses OpenRouter API with Claude 3.5 Sonnet for reliable outputs. CLI-only for focus.

## Notes for Devs
- **Customization**: Edit `prompts.json` for Architect/Director/Roleplayer logic.
- **Extensions**: Fork to add models, more agents, or param types—kept minimal as MVP.
- **Limitations**: 5 agents/25 steps; real-time but API-dependent (rate limits apply).
- **License**: MIT—free to use, modify, distribute.
