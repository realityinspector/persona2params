# main.py
import os
import json
import datetime
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

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

def get_architect_response(context):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPTS["architect"]},
            {"role": "user", "content": f"Context: {context}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty response from API for architect")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response by finding the first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = content[start_idx:end_idx + 1]
            # Clean up characters that might break JSON parsing
            import re
            # Remove control characters
            json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)
            # Escape single quotes within string values (but not structural quotes)
            # This is a simple approach - replace single quotes with escaped versions
            json_content = json_content.replace("'", "\\'")
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Failed to parse extracted JSON from architect: {json_content[:500]}...")
                raise e
        else:
            print(f"[DEBUG] No JSON found in architect response: {content}")
            raise ValueError("No valid JSON found in API response for architect")

def get_scriptwriter_response(context, characters, total_steps):
    # Create a summary of characters for the scriptwriter
    char_summary = "\n".join([f"- {char['name']}: {char['prompt']}" for char in characters])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPTS["scriptwriter"]},
            {"role": "user", "content": f"Context: {context}\nTotal conversation steps: {total_steps}\n\nCharacters:\n{char_summary}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=600,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty response from API for scriptwriter")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response by finding the first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = content[start_idx:end_idx + 1]
            # Clean up characters that might break JSON parsing
            import re
            # Remove control characters
            json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)
            # Escape single quotes within string values
            json_content = json_content.replace("'", "\\'")
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Failed to parse extracted JSON from scriptwriter: {json_content[:500]}...")
                raise e
        else:
            print(f"[DEBUG] No JSON found in scriptwriter response: {content}")
            raise ValueError("No valid JSON found in API response for scriptwriter")

def get_director_response(history, character_name, character_prompt, script, current_step, total_steps):
    current_history = "\n".join(history)

    # Format the director prompt with script and step information
    director_prompt = PROMPTS["director"].format(
        script=script,
        current_step=current_step,
        total_steps=total_steps
    )

    user_prompt = f"Conversation history:\n{current_history}\n\nNext character: {character_name}\nBase character prompt: {character_prompt}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": director_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=300,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Empty response from API for character {character_name}")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response by finding the first { and last }
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = content[start_idx:end_idx + 1]
            # Clean up control characters that might break JSON parsing
            import re
            json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Failed to parse extracted JSON for {character_name}: {json_content}")
                raise e
        else:
            print(f"[DEBUG] No JSON found in response for {character_name}: {content}")
            raise ValueError(f"No valid JSON found in API response for character {character_name}")

def get_character_response(system_prompt, history, params):
    messages = [{"role": "system", "content": system_prompt + "\n" + PROMPTS["roleplayer_suffix"]}]
    messages.extend([{"role": "assistant" if i % 2 == 0 else "user", "content": msg} for i, msg in enumerate(history)])
    for attempt in range(3):  # Retry up to 3 times
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 1.0),
            max_tokens=params.get("max_tokens", 150) + (attempt * 100),
        )
        content = response.choices[0].message.content.strip()
        if content:
            return content
    return "I see."  # Neutral fallback line for rare silences

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

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('ACT ') and ('(Step' in line or '(Steps' in line):
                # Extract step numbers from "ACT X (Steps 1-2)" or "ACT X (Step 3)"
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

                # Initialize casting for these steps
                for step in current_act_steps:
                    casting_per_step[step] = []

                # Check if CASTING is on the same line
                if 'CASTING:' in line:
                    casting_part = line.split('CASTING:')[1].strip()
                    # Remove any trailing content after the casting (like . SCENE:)
                    if 'SCENE:' in casting_part:
                        casting_part = casting_part.split('SCENE:')[0].strip()
                    elif '.' in casting_part:
                        casting_part = casting_part.split('.')[0].strip()
                    # Split by commas and clean up names
                    cast_chars = [name.strip().rstrip('.') for name in casting_part.split(',') if name.strip()]
                    # Add to all steps in current act
                    for step in current_act_steps:
                        casting_per_step[step] = cast_chars

        return casting_per_step

    casting_info = parse_casting(script)
    console.print(f"[dim]Casting parsed: {casting_info}[/dim]")

    history = [f"Setting: {setting}"]  # Initial history
    dialog = []
    director_outputs = []  # Collect Director outputs for report
    debug_infos = []  # Collect detailed debug info per step

    for step in range(n_steps):
        current_step = step + 1

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
                # Select one character from the cast for this step
                # Rotate through cast characters more frequently for better dialogue flow
                char_idx = step % len(available_chars)
                char = available_chars[char_idx]
                should_speak = True
        else:
            # No casting info, fall back to cycling through all characters
            char_idx = step % len(characters)
            char = characters[char_idx]
            should_speak = True

        name = char["name"]
        base_prompt = char["prompt"]

        # Director step
        director_json = get_director_response(history, name, base_prompt, script, step + 1, n_steps)
        updated_prompt = director_json["updated_prompt"]
        params = director_json["params"]

        # Collect Director output for report
        director_outputs.append({
            "step": step + 1,
            "character": name,
            "updated_prompt": updated_prompt,
            "params": params
        })

        if args.debug:
            print(f"\n[DEBUG] Director for {name}:")
            print(f"Updated Prompt: {updated_prompt}")
            print(f"Params: {params}")

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
            f.write(f"- **Updated Prompt:** {director_output['updated_prompt']}\n")
            f.write(f"- **Params:** {director_output['params']}\n\n")

        f.write("### Detailed Debug Info\n")
        for debug_info in debug_infos:
            f.write(f"**Step {debug_info['step']} - {debug_info['character']}:**\n")
            f.write(f"- **Director JSON:** {debug_info['director_json']}\n")
            f.write(f"- **Character Prompt:** {debug_info['character_prompt']}\n")
            f.write(f"- **Raw Response:** {debug_info['raw_response']}\n\n")

    console.print(f"\n[bold yellow]Report saved to: {report_path}[/bold yellow]")

if __name__ == "__main__":
    main()