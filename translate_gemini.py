#!/usr/bin/env python3
"""
CLI script to call Google Gemini text-generation for translation with:
- API key from .env
- dynamic keyword-based context loaded from contexts.json
- character-limit batching by line (no line splitting), controlled in Python code
- configurable number of rounds
- directory-based processing of .yml files, preserving subfolder structure
- translation of YAML values only (keys and comments are not translated)

Dependencies:
    pip install google-genai

Usage:
    python translate_gemini.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_CONTEXTS_FILE = REPO_ROOT / "contexts.json"
DEFAULT_PROMPTS_FILE = REPO_ROOT / "prompts.json"

# Character limit per Gemini request, controlled from Python code only
MAX_CHARS_PER_REQUEST = 4000

# File extension to process
YML_EXTENSION = ".yml"

# Configuration values controlled from Python code only
# Adjust these paths/values as needed.
INPUT_DIR = REPO_ROOT / "raw_data"
OUTPUT_DIR = REPO_ROOT / "translated"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
ROUNDS = 1
PROMPT_KEY = "default"

# Maximum number of API calls allowed in a single run (0 = unlimited)
MAX_API_CALLS = 1


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

_api_call_count = 0


def log(message: str) -> None:
    """Print a timestamped log message to stdout."""
    print(f"[LOG] {message}")


# ---------------------------------------------------------------------------
# Configuration loaders
# ---------------------------------------------------------------------------

def load_env_file(env_path: Path) -> None:
    """
    Minimal .env loader.
    Lines like KEY=VALUE are loaded into os.environ if not already present.
    """
    log(f"Loading .env file from: {env_path}")
    if not env_path.is_file():
        log(".env file not found — skipping")
        return

    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Warning: failed to read .env file at {env_path}: {exc}", file=sys.stderr)
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    log(".env file loaded successfully")


def load_contexts(contexts_path: Path) -> Dict[str, str]:
    """
    Load keyword -> context mapping from JSON.
    Returns an empty dict if the file is missing.
    """
    log(f"Loading contexts from: {contexts_path}")
    if not contexts_path.is_file():
        print(f"Warning: contexts file not found at {contexts_path}, continuing without extra context.", file=sys.stderr)
        return {}

    try:
        with contexts_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to load contexts from {contexts_path}: {exc}", file=sys.stderr)
        return {}

    if not isinstance(data, dict):
        print(f"Warning: contexts file {contexts_path} must contain a JSON object (keyword -> context). Ignoring.", file=sys.stderr)
        return {}

    # Ensure all keys/values are strings
    result: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        result[key] = value

    log(f"Loaded {len(result)} context keyword(s)")
    return result


def load_prompts(prompts_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load prompt configurations from JSON.

    Expected structure:
    {
      "prompt_key": {
        "system_instruction": "...",
        "contents_template": ".... {text} .... {context} ...."
      },
      ...
    }
    """
    log(f"Loading prompts from: {prompts_path}")
    if not prompts_path.is_file():
        raise RuntimeError(f"Prompts file not found at {prompts_path}")

    try:
        with prompts_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load prompts from {prompts_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Prompts file {prompts_path} must contain a JSON object.")

    prompts: Dict[str, Dict[str, str]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        system_instruction = value.get("system_instruction")
        contents_template = value.get("contents_template")
        if isinstance(system_instruction, str) and isinstance(contents_template, str):
            prompts[key] = {
                "system_instruction": system_instruction,
                "contents_template": contents_template,
            }
    if not prompts:
        raise RuntimeError(f"No valid prompt entries found in {prompts_path}")

    log(f"Loaded {len(prompts)} prompt key(s): {list(prompts.keys())}")
    return prompts


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def batch_lines_by_char_limit(lines: Sequence[str], max_chars: int) -> Iterable[List[str]]:
    """
    Group lines into batches whose total character count (including newlines
    from the original lines) does not exceed max_chars.

    - Lines are never split.
    - If a single line is longer than max_chars and the current batch is empty,
      that line is yielded as its own batch.
    """
    if max_chars <= 0:
        yield list(lines)
        return

    current_batch: List[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line)

        if not current_batch:
            # If this single line exceeds the limit, still yield it alone.
            if line_len > max_chars:
                yield [line]
                continue
            current_batch.append(line)
            current_len = line_len
            continue

        if current_len + line_len > max_chars:
            # Current batch is full; start a new one with this line.
            yield current_batch
            current_batch = [line]
            current_len = line_len
        else:
            current_batch.append(line)
            current_len += line_len

    if current_batch:
        yield current_batch


def build_context_hints(batch_text: str, contexts: Dict[str, str]) -> str:
    """
    Build a context hints string based on keyword matches in batch_text.
    Returns an empty string if no keywords match.
    """
    lower_text = batch_text.lower()

    matched_contexts: List[str] = []
    for keyword, context_text in contexts.items():
        if keyword.lower() in lower_text:
            matched_contexts.append(f"- {keyword}: {context_text}")

    if not matched_contexts:
        return ""

    return "Context hints:\n" + "\n".join(matched_contexts)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def create_gemini_client() -> genai.Client:
    """
    Initialize the Gemini client using GEMINI_API_KEY (or API_KEY) from the environment.
    """
    log("Initializing Gemini client...")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or API_KEY) is not set. "
            "Create a .env file with GEMINI_API_KEY=your_key or export it in your shell."
        )

    client = genai.Client(api_key=api_key)
    log("Gemini client initialized successfully")
    return client


def generate_translation_for_batch(
    client: genai.Client,
    model: str,
    system_instruction: str,
    contents: str,
) -> str:
    """
    Call Gemini text generation for a single batch and return the response text.
    Raises RuntimeError if the MAX_API_CALLS limit has been reached.
    """
    global _api_call_count

    if MAX_API_CALLS > 0 and _api_call_count >= MAX_API_CALLS:
        raise RuntimeError(
            f"Reached maximum API call limit ({MAX_API_CALLS}). "
            "Increase MAX_API_CALLS or set it to 0 for unlimited calls."
        )

    _api_call_count += 1
    log(f"  API call #{_api_call_count}" + (f"/{MAX_API_CALLS}" if MAX_API_CALLS > 0 else "") + f" — model: {model}")

    try:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
            contents=contents,
        )
    except Exception as exc:  # noqa: BLE001 - surface API errors to user
        print(f"Error during generation: {exc}", file=sys.stderr)
        return ""

    # response.text should contain the concatenated text of candidate parts.
    return getattr(response, "text", "") or ""


# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

def translate_segments(
    segments: Sequence[str],
    client: genai.Client,
    model: str,
    system_instruction: str,
    contents_template: str,
    contexts: Dict[str, str],
    rounds: int,
) -> List[str]:
    """
    Translate a list of text segments (one segment per YAML value).

    - Batches segments by character limit.
    - For each batch, fills the contents_template with {text} and {context}.
    - Asks the model to return exactly one output line per input line.
    - Splits the final combined output back into per-segment translations.
    """
    if not segments:
        return []

    # Prepare lines with explicit newlines for batching
    lines_for_batching = [segment + "\n" for segment in segments]

    batch_outputs: List[str] = []
    batches = list(batch_lines_by_char_limit(lines_for_batching, max_chars=MAX_CHARS_PER_REQUEST))

    log(f"  Translating {len(segments)} segment(s) across {len(batches)} batch(es) | rounds={rounds}")

    for idx, batch_lines in enumerate(batches, start=1):
        log(f"  Batch {idx}/{len(batches)} — {len(batch_lines)} line(s), ~{sum(len(l) for l in batch_lines)} chars")
        batch_text = "".join(batch_lines)
        context_hints = build_context_hints(batch_text, contexts)

        if context_hints:
            log(f"  Context hints matched for batch {idx}")

        try:
            contents = contents_template.format(text=batch_text, context=context_hints)
        except Exception as exc:  # noqa: BLE001
            print(f"Error formatting contents_template: {exc}", file=sys.stderr)
            contents = batch_text

        last_output = ""
        for round_num in range(1, rounds + 1):
            log(f"  Batch {idx}/{len(batches)} — round {round_num}/{rounds}")
            last_output = generate_translation_for_batch(
                client=client,
                model=model,
                system_instruction=system_instruction,
                contents=contents,
            )

        batch_outputs.append(last_output)

    combined_output = "".join(batch_outputs)
    translated_lines = combined_output.splitlines()

    if len(translated_lines) != len(segments):
        print(
            f"Warning: expected {len(segments)} translated lines, "
            f"but got {len(translated_lines)}. Adjusting to match.",
            file=sys.stderr,
        )
    # Adjust length to match original segments
    result: List[str] = []
    for idx, original in enumerate(segments):
        if idx < len(translated_lines):
            result.append(translated_lines[idx])
        else:
            # Fallback: keep original if missing translation
            result.append(original)
    return result


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

YAML_VALUE_PATTERN = re.compile(r'^(\s*[^#\s][^:]*:\s*)"(.*)"(.*)$')


def extract_yaml_values(lines: Sequence[str]) -> Tuple[List[str], List[Tuple[int, str, str, str]]]:
    """
    Extract quoted YAML values from lines.

    Returns:
    - values: list of strings inside quotes
    - metadata: list of tuples (line_index, prefix, value, suffix_plus_newline)
    """
    values: List[str] = []
    metadata: List[Tuple[int, str, str, str]] = []

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        # Skip full-line comments
        if stripped.startswith("#"):
            continue

        # Preserve original line ending
        line_ending = ""
        if line.endswith("\r\n"):
            line_ending = "\r\n"
            core = line[:-2]
        elif line.endswith("\n"):
            line_ending = "\n"
            core = line[:-1]
        else:
            core = line

        match = YAML_VALUE_PATTERN.match(core)
        if not match:
            continue

        prefix, value, suffix = match.groups()
        values.append(value)
        metadata.append((idx, prefix, value, suffix + line_ending))

    return values, metadata


def apply_translations_to_lines(
    lines: List[str],
    metadata: Sequence[Tuple[int, str, str, str]],
    translated_values: Sequence[str],
) -> List[str]:
    """
    Apply translated values back into the original YAML lines.
    """
    new_lines = list(lines)
    for i, meta in enumerate(metadata):
        if i >= len(translated_values):
            break
        line_index, prefix, _original_value, suffix_plus_newline = meta
        new_value = translated_values[i]
        new_lines[line_index] = f'{prefix}"{new_value}"{suffix_plus_newline}'
    return new_lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    log("=== Stellaris Mod Thai Translator ===")
    log(f"MAX_API_CALLS = {MAX_API_CALLS if MAX_API_CALLS > 0 else 'unlimited'}")
    log(f"ROUNDS        = {ROUNDS}")
    log(f"MODEL         = {MODEL_NAME}")
    log(f"INPUT_DIR     = {INPUT_DIR}")
    log(f"OUTPUT_DIR    = {OUTPUT_DIR}")

    # Load configuration files first
    load_env_file(DEFAULT_ENV_FILE)
    contexts = load_contexts(DEFAULT_CONTEXTS_FILE)
    try:
        prompts = load_prompts(DEFAULT_PROMPTS_FILE)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    input_dir = Path(INPUT_DIR)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory ready: {output_dir}")

    prompt_key = PROMPT_KEY
    prompt_cfg = prompts.get(prompt_key)
    if not prompt_cfg:
        print(f"Error: prompt key '{prompt_key}' not found in {DEFAULT_PROMPTS_FILE}", file=sys.stderr)
        return 0

    log(f"Using prompt key: '{prompt_key}'")

    try:
        client = create_gemini_client()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rounds = max(1, int(ROUNDS))
    system_instruction = prompt_cfg["system_instruction"]
    contents_template = prompt_cfg["contents_template"]

    # Collect all .yml files to process
    all_yml_files = [p for p in input_dir.rglob(f"*{YML_EXTENSION}") if p.is_file()]
    log(f"Found {len(all_yml_files)} .yml file(s) in {input_dir}")

    skipped = 0
    processed = 0
    errors = 0

    # Walk through input_dir, process all .yml files, and mirror structure in output_dir
    for file_num, path in enumerate(all_yml_files, start=1):
        rel_path = path.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Skip check: if the output file already exists, skip translation ---
        if output_path.exists():
            log(f"[{file_num}/{len(all_yml_files)}] SKIP (already translated): {rel_path}")
            skipped += 1
            continue

        log(f"[{file_num}/{len(all_yml_files)}] Processing: {rel_path}")

        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as exc:
            print(f"Error: failed to read input file {path}: {exc}", file=sys.stderr)
            errors += 1
            continue

        log(f"  Read {len(lines)} line(s) from {path.name}")

        values, metadata = extract_yaml_values(lines)

        if not values:
            # No translatable values; just copy the file as-is
            log(f"  No translatable values found — copying file as-is")
            try:
                output_path.write_text("".join(lines), encoding="utf-8")
                log(f"  Wrote (copy): {output_path}")
            except OSError as exc:
                print(f"Error: failed to write output file {output_path}: {exc}", file=sys.stderr)
                errors += 1
            processed += 1
            continue

        log(f"  Found {len(values)} translatable value(s)")

        try:
            translated_values = translate_segments(
                segments=values,
                client=client,
                model=MODEL_NAME,
                system_instruction=system_instruction,
                contents_template=contents_template,
                contexts=contexts,
                rounds=rounds,
            )
        except RuntimeError as exc:
            # MAX_API_CALLS limit hit
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        new_lines = apply_translations_to_lines(lines, metadata, translated_values)

        try:
            output_path.write_text("".join(new_lines), encoding="utf-8")
            log(f"  Wrote translated file: {output_path}")
        except OSError as exc:
            print(f"Error: failed to write output file {output_path}: {exc}", file=sys.stderr)
            errors += 1
            continue

        processed += 1

    log("=== Translation run complete ===")
    log(f"  Files processed : {processed}")
    log(f"  Files skipped   : {skipped} (already in translated/)")
    log(f"  Errors          : {errors}")
    log(f"  Total API calls : {_api_call_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
