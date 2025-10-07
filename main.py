# main.py
import os
import json
import datetime
import argparse
import statistics
import time
import re
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, Timeout
from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL = "openai/gpt-4o"  # Stricter adherence than Claude for roleplaying

# Initialize Rich console for styled output
console = Console()

# Character color mapping (cycling through distinct colors)
CHARACTER_COLORS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "bright_red", "bright_blue", "cyan", "green", "yellow", "magenta"
]

# Load prompts from prompts.json
with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)

def robust_llm_call(messages, response_format=None, temperature=0.7, max_tokens=500,
                    max_retries=3, fallback_response=None):
    """Robust LLM call with retry logic, error handling, and JSON parsing."""
    last_error = None

    for attempt in range(max_retries):
        try:
            # Add error context for retries
            if attempt > 0 and last_error:
                error_context = f"\n\nPrevious attempt failed with: {str(last_error)[:100]}... Please ensure valid JSON format."
                if isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += error_context

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format=response_format,
                temperature=min(temperature + attempt * 0.1, 1.0),  # Slightly increase temperature on retries
                max_tokens=max_tokens,
                timeout=30
            )

            content = response.choices[0].message.content
            if not content or not content.strip():
                last_error = "Empty response from API"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return fallback_response or {"error": "Empty response after retries"}

            # Try to parse JSON if expected
            if response_format and response_format.get("type") == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Attempt to extract and clean JSON
                    json_content = extract_json_from_response(content)
                    if json_content:
                        try:
                            return json.loads(json_content)
                        except json.JSONDecodeError:
                            pass

                    last_error = "JSON parsing failed"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return fallback_response or {"error": "JSON parsing failed after retries", "raw_content": content[:500]}

            return content

        except (APIError, RateLimitError, Timeout) as e:
            last_error = f"API Error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                console.print(f"[dim yellow]API error, retrying in {wait_time}s...[/dim yellow]")
                time.sleep(wait_time)
                continue
            return fallback_response or {"error": f"API error after retries: {str(e)}"}

        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return fallback_response or {"error": f"Unexpected error after retries: {str(e)}"}

    return fallback_response or {"error": "All retry attempts failed"}

def extract_json_from_response(content):
    """Extract JSON from potentially malformed LLM response."""
    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)

    # Find JSON boundaries
    start_idx = content.find('{')
    end_idx = content.rfind('}')

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    json_content = content[start_idx:end_idx + 1]

    # Clean up common issues
    json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)  # Remove control chars
    json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
    json_content = re.sub(r',\s*]', ']', json_content)

    # Try to balance braces
    open_braces = json_content.count('{')
    close_braces = json_content.count('}')
    if open_braces > close_braces:
        json_content += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        json_content = '{' * (close_braces - open_braces) + json_content

    return json_content

def get_character_color(character_name, characters):
    """Get a unique color for each character"""
    char_names = [char['name'] for char in characters]
    try:
        index = char_names.index(character_name)
        return CHARACTER_COLORS[index % len(CHARACTER_COLORS)]
    except ValueError:
        return "white"  # fallback

def print_styled_dialog(character_name, dialog_text, characters):
    """Print styled dialog with colored character names"""
    color = get_character_color(character_name, characters)
    char_text = Text(f"{character_name}:", style=f"bold {color}")
    dialog_text_obj = Text(f" {dialog_text}", style="white")
    console.print(char_text, dialog_text_obj)

def analyze_param_diversity(param_history):
    """Analyze parameter diversity and variation across the conversation"""
    if not param_history:
        return "No parameter data to analyze."

    # Extract parameter values
    temperatures = [p["temperature"] for p in param_history]
    top_ps = [p["top_p"] for p in param_history]
    max_tokens_list = [p["max_tokens"] for p in param_history]

    # Calculate statistics for each parameter
    def analyze_param(values, param_name):
        if not values:
            return {}

        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        stdev_val = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        # Calculate coefficient of variation (stdev/mean) as percentage
        cv_percent = (stdev_val / mean_val * 100) if mean_val != 0 else 0

        # Count values in different ranges
        if param_name == "temperature":
            ranges = {"<0.5": 0, "0.5-0.8": 0, "0.8-1.0": 0, "1.0-1.2": 0, ">1.2": 0}
            for v in values:
                if v < 0.5: ranges["<0.5"] += 1
                elif v < 0.8: ranges["0.5-0.8"] += 1
                elif v < 1.0: ranges["0.8-1.0"] += 1
                elif v < 1.2: ranges["1.0-1.2"] += 1
                else: ranges[">1.2"] += 1
        elif param_name == "top_p":
            ranges = {"<0.5": 0, "0.5-0.8": 0, "0.8-0.95": 0, "0.95-1.0": 0, ">1.0": 0}
            for v in values:
                if v < 0.5: ranges["<0.5"] += 1
                elif v < 0.8: ranges["0.5-0.8"] += 1
                elif v < 0.95: ranges["0.8-0.95"] += 1
                elif v <= 1.0: ranges["0.95-1.0"] += 1
                else: ranges[">1.0"] += 1
        else:  # max_tokens
            ranges = {"<100": 0, "100-150": 0, "150-200": 0, "200-300": 0, ">300": 0}
            for v in values:
                if v < 100: ranges["<100"] += 1
                elif v <= 150: ranges["100-150"] += 1
                elif v <= 200: ranges["150-200"] += 1
                elif v <= 300: ranges["200-300"] += 1
                else: ranges[">300"] += 1

        return {
            "mean": round(mean_val, 3),
            "median": round(median_val, 3),
            "stdev": round(stdev_val, 3),
            "cv_percent": round(cv_percent, 1),
            "min": min_val,
            "max": max_val,
            "range": range_val,
            "ranges": ranges,
            "unique_values": len(set(values))
        }

    temp_stats = analyze_param(temperatures, "temperature")
    top_p_stats = analyze_param(top_ps, "top_p")
    max_tokens_stats = analyze_param(max_tokens_list, "max_tokens")

    # Create a rich table for the metrics
    table = Table(title="🎯 Parameter Diversity Analysis", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Mean", style="green")
    table.add_column("Median", style="green")
    table.add_column("Std Dev", style="yellow")
    table.add_column("CV %", style="yellow")
    table.add_column("Range", style="red")
    table.add_column("Unique Values", style="blue")
    table.add_column("Distribution", style="magenta")

    # Add rows for each parameter
    def format_ranges(ranges):
        return "\n".join([f"{k}: {v}" for k, v in ranges.items()])

    table.add_row(
        "Temperature",
        str(temp_stats["mean"]),
        str(temp_stats["median"]),
        str(temp_stats["stdev"]),
        str(temp_stats["cv_percent"]),
        f"{temp_stats['min']}-{temp_stats['max']}",
        str(temp_stats["unique_values"]),
        format_ranges(temp_stats["ranges"])
    )

    table.add_row(
        "Top-P",
        str(top_p_stats["mean"]),
        str(top_p_stats["median"]),
        str(top_p_stats["stdev"]),
        str(top_p_stats["cv_percent"]),
        f"{top_p_stats['min']}-{top_p_stats['max']}",
        str(top_p_stats["unique_values"]),
        format_ranges(top_p_stats["ranges"])
    )

    table.add_row(
        "Max Tokens",
        str(max_tokens_stats["mean"]),
        str(max_tokens_stats["median"]),
        str(max_tokens_stats["stdev"]),
        str(max_tokens_stats["cv_percent"]),
        f"{max_tokens_stats['min']}-{max_tokens_stats['max']}",
        str(max_tokens_stats["unique_values"]),
        format_ranges(max_tokens_stats["ranges"])
    )

    # Calculate step-to-step variation
    temp_changes = []
    top_p_changes = []
    max_tokens_changes = []

    for i in range(1, len(param_history)):
        temp_changes.append(abs(param_history[i]["temperature"] - param_history[i-1]["temperature"]))
        top_p_changes.append(abs(param_history[i]["top_p"] - param_history[i-1]["top_p"]))
        max_tokens_changes.append(abs(param_history[i]["max_tokens"] - param_history[i-1]["max_tokens"]))

    if temp_changes:
        avg_temp_change = statistics.mean(temp_changes)
        avg_top_p_change = statistics.mean(top_p_changes)
        avg_max_tokens_change = statistics.mean(max_tokens_changes)

        variation_table = Table(title="📈 Step-to-Step Variation", show_header=True, header_style="bold cyan")
        variation_table.add_column("Parameter", style="cyan")
        variation_table.add_column("Avg Step Change", style="green")
        variation_table.add_column("Max Step Change", style="red")

        variation_table.add_row(
            "Temperature",
            f"{avg_temp_change:.3f}",
            f"{max(temp_changes):.3f}"
        )
        variation_table.add_row(
            "Top-P",
            f"{avg_top_p_change:.3f}",
            f"{max(top_p_changes):.3f}"
        )
        variation_table.add_row(
            "Max Tokens",
            f"{avg_max_tokens_change:.1f}",
            f"{max(max_tokens_changes):.1f}"
        )

        return Panel(Group(table, variation_table), title="🔬 Parameter Diversity Metrics", border_style="blue")
    else:
        return Panel(table, title="🔬 Parameter Diversity Metrics", border_style="blue")

def get_architect_response(context):
    """Get architect response with robust error handling."""
    fallback = {
        "setting": "A generic setting based on the context",
        "characters": [
            {"name": "Character 1", "prompt": "A character in this scenario"},
            {"name": "Character 2", "prompt": "Another character in this scenario"}
        ]
    }

    result = robust_llm_call(
        messages=[
            {"role": "system", "content": PROMPTS["architect"]},
            {"role": "user", "content": f"Context: {context}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=500,
        fallback_response=fallback
    )

    if isinstance(result, dict) and "error" in result:
        console.print(f"[dim yellow]Architect fallback used: {result['error']}[/dim yellow]")
        return fallback

    return result

def get_scriptwriter_response(context, characters, total_steps):
    """Get scriptwriter response with robust error handling."""
    char_summary = "\n".join([f"- {char['name']}: {char['prompt']}" for char in characters])

    fallback = {
        "script": f"ACT 1 (Steps 1-{total_steps//2}): Introduction and setup of the scenario.\n\nACT 2 (Steps {total_steps//2 + 1}-{total_steps}): Development and resolution of the narrative."
    }

    result = robust_llm_call(
        messages=[
            {"role": "system", "content": PROMPTS["scriptwriter"]},
            {"role": "user", "content": f"Context: {context}\nTotal conversation steps: {total_steps}\n\nCharacters:\n{char_summary}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=600,
        fallback_response=fallback
    )

    if isinstance(result, dict) and "error" in result:
        console.print(f"[dim yellow]Scriptwriter fallback used: {result['error']}[/dim yellow]")
        return fallback

    return result

def get_director_response(history, character_name, character_prompt, script, current_step, total_steps, current_pattern, pattern_position):
    """Get director response with robust error handling."""
    current_history = "\n".join(history)
    director_prompt = PROMPTS["director"].format(
        script=script,
        current_step=current_step,
        total_steps=total_steps
    )

    user_prompt = f"Conversation history:\n{current_history}\n\nNext character: {character_name}\nBase character prompt: {character_prompt}\n\nCURRENT PATTERN: {current_pattern} | POSITION: {pattern_position}"

    fallback = {
        "pattern": current_pattern,
        "pattern_position": pattern_position,
        "updated_prompt": f"{character_name} continues the conversation naturally based on their personality and the current scene.",
        "params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 150}
    }

    result = robust_llm_call(
        messages=[
            {"role": "system", "content": director_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=300,
        fallback_response=fallback
    )

    if isinstance(result, dict) and "error" in result:
        console.print(f"[dim yellow]Director fallback used for {character_name}: {result['error']}[/dim yellow]")
        return fallback

    return result

def get_character_response(system_prompt, history, params):
    """Get character response with robust error handling."""
    messages = [{"role": "system", "content": system_prompt + "\n" + PROMPTS["roleplayer_suffix"]}]
    messages.extend([{"role": "assistant" if i % 2 == 0 else "user", "content": msg} for i, msg in enumerate(history)])

    # Increase max_tokens on retries for potentially stuck responses
    base_max_tokens = params.get("max_tokens", 150)

    result = robust_llm_call(
        messages=messages,
        response_format=None,  # Not expecting JSON for character responses
        temperature=params.get("temperature", 0.7),
        max_tokens=base_max_tokens,
        max_retries=3,
        fallback_response="I see."  # Neutral fallback line
    )

    if isinstance(result, dict) and "error" in result:
        console.print(f"[dim yellow]Character response fallback used: {result['error']}[/dim yellow]")
        return "I see."

    return result.strip() if result else "I see."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    context = input("Enter the context (e.g., 'shakespeare's midsummer nights dream'): ").strip()
    n_steps = int(input("Enter number of dialog steps (max 25): "))
    if n_steps > 25:
        n_steps = 25

    # Architect sets up characters
    architect_json = get_architect_response(context)
    setting = architect_json["setting"]
    characters = architect_json["characters"]  # List of dicts: {"name": str, "prompt": str}
    if len(characters) > 5:
        characters = characters[:5]

    console.print(f"[bold blue]Setting:[/bold blue] {setting}")
    console.print("[bold blue]Characters:[/bold blue]")
    for char in characters:
        color = get_character_color(char['name'], characters)
        console.print(f"  • [{color}]{char['name']}[/{color}]: {char['prompt']}")

    # Scriptwriter creates narrative outline
    scriptwriter_json = get_scriptwriter_response(context, characters, n_steps)
    script = scriptwriter_json["script"]
    console.print(f"\n[bold green]Narrative Script:[/bold green]")
    console.print(Panel(script, border_style="green"))

    # Parse casting information from script
    def parse_casting(script_text):
        """Parse casting information from script to determine which characters participate in each step"""
        casting_per_step = {}
        lines = script_text.split('\n')
        current_act_steps = []

        if args.debug:
            console.print(f"[dim cyan]DEBUG: Parsing {len(lines)} script lines[/dim cyan]")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('ACT ') and ('(Step' in line or '(Steps' in line):
                # Extract step numbers from "ACT X (Step 1)" or "ACT X (Steps 1-2)"
                import re
                if '(Steps ' in line:
                    step_match = re.search(r'ACT \d+ \(Steps (\d+)-(\d+)\)', line)
                    if step_match:
                        start_step = int(step_match.group(1))
                        end_step = int(step_match.group(2))
                        current_act_steps = list(range(start_step, end_step + 1))
                elif '(Step ' in line:
                    step_match = re.search(r'ACT \d+ \(Step (\d+)\)', line)
                    if step_match:
                        step_num = int(step_match.group(1))
                        current_act_steps = [step_num]

                if args.debug:
                    console.print(f"[dim cyan]Found ACT with steps: {current_act_steps}[/dim cyan]")

                # Initialize casting for these steps
                for step in current_act_steps:
                    casting_per_step[step] = []

                # Check for inline CASTING in the ACT line
                if 'CASTING:' in line:
                    casting_part = line.split('CASTING:')[1].strip()
                    # Remove any trailing content (like "SCENE:")
                    if 'SCENE:' in casting_part:
                        casting_part = casting_part.split('SCENE:')[0].strip()
                    elif '.' in casting_part:
                        casting_part = casting_part.split('.')[0].strip()
                    # Split by commas and clean up names
                    cast_chars = [name.strip().rstrip('.') for name in casting_part.split(',') if name.strip()]
                    # Add to all steps in current act
                    for step in current_act_steps:
                        casting_per_step[step] = cast_chars

                    if args.debug:
                        console.print(f"[dim cyan]Parsed casting for steps {current_act_steps}: {cast_chars}[/dim cyan]")

            i += 1

        return casting_per_step

    casting_info = parse_casting(script)
    console.print(f"[dim]Casting parsed: {casting_info}[/dim]")

    # Debug: show script lines for casting parsing
    if args.debug:
        console.print(f"[dim yellow]Script lines for debugging:[/dim yellow]")
        for i, line in enumerate(script.split('\n')[:20]):  # First 20 lines
            if 'CASTING' in line or line.strip().startswith('ACT '):
                console.print(f"[dim yellow]{i}: {line.strip()}[/dim yellow]")

    history = [f"Setting: {setting}"]  # Initial history
    dialog = []
    director_outputs = []  # Collect Director outputs for report
    debug_infos = []  # Collect detailed debug info per step
    param_history = []  # Collect parameter values for analysis

    # Track pattern state across conversation
    pattern_tracker = {}  # Track turn counts for each pattern type

    def get_current_pattern_for_step(step_num, script_text):
        """Determine which pattern should be used for the current step based on script."""
        lines = script_text.split('\n')
        current_act_pattern = None

        # Find the act that contains this step and its pattern
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this is an ACT line that contains our step
            if line.startswith('ACT ') and ('(Steps ' in line or '(Step ' in line):
                act_contains_step = False

                if '(Steps ' in line:
                    import re
                    step_match = re.search(r'ACT \d+ \(Steps (\d+)-(\d+)\)', line)
                    if step_match:
                        start_step = int(step_match.group(1))
                        end_step = int(step_match.group(2))
                        if start_step <= step_num <= end_step:
                            act_contains_step = True
                elif '(Step ' in line:
                    step_match = re.search(r'ACT \d+ \(Step (\d+)\)', line)
                    if step_match and int(step_match.group(1)) == step_num:
                        act_contains_step = True

                # If this act contains our step, find its pattern
                if act_contains_step:
                    # Look for PATTERN in subsequent lines of this act
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith('ACT '):
                        pattern_line = lines[j].strip()
                        if 'PATTERN:' in pattern_line:
                            # Extract pattern from line like "PATTERN: ARGUMENT→CONFESSION."
                            pattern_part = pattern_line.split('PATTERN:')[1].strip()
                            if '.' in pattern_part:
                                pattern_part = pattern_part.split('.')[0].strip()
                            return pattern_part
                        j += 1
                    break

            i += 1

        return "GENERAL"  # fallback

    for step in range(n_steps):
        current_step = step + 1

        # Determine current pattern for this step
        current_pattern = get_current_pattern_for_step(current_step, script)

        # Check if this step has specific casting
        if current_step in casting_info and casting_info[current_step]:
            # Only use characters that are cast for this step
            cast_names = casting_info[current_step]
            available_chars = [char for char in characters if char['name'] in cast_names]

            if not available_chars:
                # No cast characters available, use first character but make them silent
                char = characters[0]
                should_speak = False
            else:
                # Select one character to speak per step, with pattern-aware rotation
                pattern_turn = pattern_tracker.get(current_pattern, 0)

                if current_pattern == 'CONFESSION' and 'Hermia' in cast_names:
                    # For CONFESSION pattern, Hermia should speak first (the confessor)
                    if pattern_turn == 0:
                        char = next(char for char in available_chars if char['name'] == 'Hermia')
                    else:
                        # Then alternate between confessor and supporter
                        char_idx = pattern_turn % len(available_chars)
                        char = available_chars[char_idx]
                else:
                    # Default rotation through cast
                    char_idx = pattern_turn % len(available_chars)
                    char = available_chars[char_idx]
                should_speak = True
        else:
            # No casting info, fall back to cycling through all characters
            char_idx = step % len(characters)
            char = characters[char_idx]
            should_speak = True

        name = char["name"]
        base_prompt = char["prompt"]

        # Determine current pattern for this step
        current_pattern = get_current_pattern_for_step(current_step, script)

        # Track pattern position
        if current_pattern not in pattern_tracker:
            pattern_tracker[current_pattern] = 0
        pattern_tracker[current_pattern] += 1
        pattern_position = f"Turn {pattern_tracker[current_pattern]}"

        # Director step
        director_json = get_director_response(history, name, base_prompt, script, step + 1, n_steps, current_pattern, pattern_position)
        pattern = director_json.get("pattern", "GENERAL")
        pattern_position = director_json.get("pattern_position", "Turn 1")
        updated_prompt = director_json["updated_prompt"]
        params = director_json["params"]

        # Collect params for analysis
        param_history.append({
            "step": step + 1,
            "character": name,
            "pattern": pattern,
            "pattern_position": pattern_position,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0),
            "max_tokens": params.get("max_tokens", 150)
        })

        # Collect Director output for report
        director_outputs.append({
            "step": step + 1,
            "character": name,
            "pattern": pattern,
            "pattern_position": pattern_position,
            "updated_prompt": updated_prompt,
            "params": params
        })

        if args.debug:
            console.print(f"\n[bold cyan][DEBUG] Director for {name}:[/bold cyan]")
            console.print(f"[cyan]Pattern: {pattern} | Position: {pattern_position}[/cyan]")
            console.print(f"[cyan]Updated Prompt: {updated_prompt}[/cyan]")
            console.print(f"[cyan]Params: {params}[/cyan]")

        # Character response with enhanced retry logic
        # Check if character is cast (allow partial name matching for first names)
        is_cast = False
        if current_step in casting_info:
            cast_names = casting_info[current_step]
            # Check for exact match first, then try first name matching
            if name in cast_names:
                is_cast = True
            else:
                # Try matching first names (e.g., "Romeo Montague" matches "Romeo")
                first_name = name.split()[0] if name.split() else name
                is_cast = any(first_name in cast_name for cast_name in cast_names)
        if args.debug:
            console.print(f"[dim cyan][DEBUG] Step {current_step}: {name} | should_speak: {should_speak} | in_casting: {is_cast} | cast_list: {casting_info.get(current_step, [])}[/dim cyan]")

        if should_speak and is_cast:
            # Character is cast for this step, they should speak
            response = get_character_response(updated_prompt, history, params)
        else:
            # Character is not cast for this step, they remain silent
            response = "..."

        # Post-process response to strip actions and prefixes (only if they spoke)
        if response != "...":
            import re
            response = re.sub(r'\*.*?\*', '', response)  # Strip *actions*
            response = re.sub(r'^.*?:', '', response).strip()  # Strip any "Name:" prefixes

        full_line = f"{name}: {response}"
        dialog.append(full_line)
        history.append(full_line)

        # Collect detailed debug info for report
        debug_infos.append({
            "step": step + 1,
            "character": name,
            "director_json": {"updated_prompt": updated_prompt, "params": params},
            "character_prompt": updated_prompt + '\n' + PROMPTS['roleplayer_suffix'],
            "raw_response": response
        })

        if args.debug:
            console.print(f"[dim cyan][DEBUG] Character Prompt: {updated_prompt + '\n' + PROMPTS['roleplayer_suffix']}[/dim cyan]")
            console.print(f"[dim cyan][DEBUG] Response: {response}[/dim cyan]")
        else:
            print_styled_dialog(name, response, characters)

    # Generate report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Dialog Report\n\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")
        f.write(f"**Context:** {context}\n\n")
        f.write(f"**Narrative Script:** {script}\n\n")
        f.write("## Dialog\n\n")
        f.write("\n".join(dialog))
        f.write("\n\n## Prompts\n\n")
        f.write("### Architect Prompt\n")
        f.write(PROMPTS["architect"] + "\n\n")
        f.write("### Director Prompt\n")
        f.write(PROMPTS["director"] + "\n\n")
        f.write("### Roleplayer Suffix\n")
        f.write(PROMPTS["roleplayer_suffix"] + "\n\n")
        f.write("### Character Prompts\n")
        for char in characters:
            f.write(f"**{char['name']}:** {char['prompt']}\n\n")
        f.write("### Director Outputs\n")
        for director_output in director_outputs:
            f.write(f"**Step {director_output['step']} - {director_output['character']}:**\n")
            f.write(f"- **Pattern:** {director_output['pattern']} | **Position:** {director_output['pattern_position']}\n")
            f.write(f"- **Updated Prompt:** {director_output['updated_prompt']}\n")
            f.write(f"- **Params:** {director_output['params']}\n\n")

        f.write("### Detailed Debug Info\n")
        for debug_info in debug_infos:
            f.write(f"**Step {debug_info['step']} - {debug_info['character']}:**\n")
            f.write(f"- **Director JSON:** {debug_info['director_json']}\n")
            f.write(f"- **Character Prompt:** {debug_info['character_prompt']}\n")
            f.write(f"- **Raw Response:** {debug_info['raw_response']}\n\n")

    # Display parameter diversity metrics in debug mode
    if args.debug and param_history:
        console.print("\n" + "="*80)
        console.print(analyze_param_diversity(param_history))
        console.print("="*80 + "\n")

    console.print(f"\n[bold yellow]Report saved to: {report_path}[/bold yellow]")

if __name__ == "__main__":
    main()