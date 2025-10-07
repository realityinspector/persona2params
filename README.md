# Persona2Params

## TL;DR
A Python CLI tool for simulating multi-agent roleplay dialogs with up to 5 LLM characters. It uses a 3-stage pipeline: Architect creates characters, Scriptwriter crafts narrative arcs, Director coordinates per-step LLM parameters (temperature, top_p, max_tokens) for evolving emotional and narrative behaviors. OSS MVP using OpenRouter API—easy to run, extend, and debug for AI devs and storytellers.

## Overview: Script-Driven Persona-to-Params Mapping for Narrative Roleplay
Persona2Params is an open-source MVP that brings static character descriptions to life through script-guided LLM parameter tuning. It uses a 3-agent system:

1. **Architect**: Creates characters and setting from context
2. **Scriptwriter**: Crafts comprehensive narrative arcs (setup → rising action → climax → resolution)
3. **Director**: Coordinates per-step parameters, ensuring characters advance the story rather than just converse

This approach creates **coherent narratives** where characters evolve through emotional arcs and story progression, not just reactive dialogue. Built as a monolith for simplicity, it's ideal for AI-driven storytelling, game prototyping, or behavioral simulations. Contributions welcome to refine or expand.

## Technical Deep Dive: 3-Agent Script-Driven Pipeline
### 1. Architect: Character & Setting Creation
- **Input**: Context string (e.g., "Midsummer Night's Dream on Mars")
- **Output**: 5 characters with personas + setting description

### 2. Scriptwriter: Narrative Arc Design
- **Input**: Context + character list
- **Output**: Complete story outline (ACT 1-4: setup → rising action → climax → resolution)
- **Purpose**: Provides narrative roadmap for coherent storytelling

### 3. Director: Per-Step Coordination
Reviews conversation history, base persona, script position, then outputs:
```json
{
  "updated_prompt": "What character should express next in the story arc",
  "params": {
    "temperature": 0.8,    // Higher for chaotic/drunk states (>1.0 possible)
    "top_p": 0.95,         // Higher for focused/military precision
    "max_tokens": 150      // Tuned for response length
  }
}
```

### 4. Character: Contextual Response
Combines Director's prompt with roleplayer instructions for pure dialog output.

### 5. Dynamic Evolution Through Narrative
Same persona evolves through story phases:
- **ACT 1** (Setup): Establishing character, relationships
- **ACT 2** (Rising Action): Building tension, conflicts emerge
- **ACT 3** (Climax): Peak emotional/behavioral intensity
- **ACT 4** (Resolution): Character growth, story conclusion

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
