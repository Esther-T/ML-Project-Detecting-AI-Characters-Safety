"""
annotate.py
-----------
Usage:
    python annotate.py            # full run
    python annotate.py --dry-run  # preview 3 rows, no API calls
    python annotate.py --status   # show quota, then exit
"""

import argparse
import json
import os
import time
from datetime import date

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()

# ============================================================
# CONFIG — single key, single model
# ============================================================

API_KEY = os.getenv("GEMINI_KEY", "").strip()
MODEL   = "gemini-3.1-flash-lite-preview"

if not API_KEY:
    raise ValueError("No API key found. Set GEMINI_KEY in your .env file.")

FEATHER_INPUT    = "ml_df_part1.feather"
FEATHER_OUTPUT   = "annotated_part1.feather"
QUOTA_LOG        = "quota_log.json"
TAXONOMY_DIR     = "taxonomies"
PROMPT_DIR       = "prompts"
CHECKPOINT_EVERY = 10
MAX_RETRIES      = 3
DRY_RUN_ROWS     = 3

SLEEP_BETWEEN_DIMS = 2   # seconds between each of the 6 dimension calls
SLEEP_BETWEEN_ROWS = 5  # seconds between rows
SLEEP_ON_ERROR     = 10  # seconds after a non-429 error

ANNOTATION_COLS = [
    "demographic", "occupation", "space",
    "relationship", "favorability", "personality"
]

# ============================================================
# Quota tracker — single key version
# ============================================================

def load_quota_log():
    today = str(date.today())
    if os.path.exists(QUOTA_LOG):
        with open(QUOTA_LOG) as f:
            log = json.load(f)
        if log.get("date") != today:
            log = {"date": today, "usage": 0, "exhausted": False}
    else:
        log = {"date": today, "usage": 0, "exhausted": False}
    return log

def save_quota_log(log):
    with open(QUOTA_LOG, "w") as f:
        json.dump(log, f, indent=2)

def increment_usage(log):
    log["usage"] += 1
    save_quota_log(log)

def mark_exhausted(log):
    log["exhausted"] = True
    save_quota_log(log)

def print_quota_status(log):
    status = "EXHAUSTED" if log["exhausted"] else "ok"
    print(f"\n{'─'*40}")
    print(f"  Quota — {log['date']}")
    print(f"  Key    : ...{API_KEY[-6:]}  [{MODEL}]  {status}")
    print(f"  Calls  : {log['usage']} used today")
    print(f"{'─'*40}\n")

# ============================================================
# Load data, taxonomies, prompts
# ============================================================

def load_data():
    if os.path.exists(FEATHER_OUTPUT):
        df = pd.read_feather(FEATHER_OUTPUT)
        print(f"Resuming: {FEATHER_OUTPUT}")
    else:
        df = pd.read_feather(FEATHER_INPUT)
        for col in ANNOTATION_COLS:
            df[col] = None
        print(f"Fresh start — {len(df)} rows from {FEATHER_INPUT}")
    return df

def load_taxonomies():
    yaml_files = {}
    for fn in sorted(os.listdir(TAXONOMY_DIR)):
        if fn.endswith(".yaml"):
            with open(os.path.join(TAXONOMY_DIR, fn)) as f:
                yaml_files[fn] = yaml.safe_load(f)
    return {
        "demographic":  yaml_files["VII_Demographic.yaml"],
        "occupation":   yaml_files["VIII_Occupation.yaml"],
        "space":        yaml_files["IX_Space.yaml"],
        "relationship": yaml_files["X_Relationship.yaml"],
        "personality":  yaml_files["XI_Personality.yaml"],
    }

def load_prompts():
    prompt_files = {}
    for fn in sorted(os.listdir(PROMPT_DIR)):
        if fn.endswith(".txt"):
            with open(os.path.join(PROMPT_DIR, fn)) as f:
                prompt_files[fn] = f.read()
    return {
        "demographic":  prompt_files["demographic_prompt.txt"],
        "occupation":   prompt_files["occupation_prompt.txt"],
        "space":        prompt_files["space_prompt.txt"],
        "relationship": prompt_files["relationship_prompt.txt"],
        "favorability": prompt_files["favorability_prompt.txt"],
        "personality":  prompt_files["personality_prompt.txt"],
    }

# ============================================================
# LLM call — single key, no rotation
# ============================================================

client = genai.Client(api_key=API_KEY)

def prepare_character_input(row):
    return yaml.dump({
        "tags":        row["tags"],
        "description": row["description"] if pd.notna(row["description"]) else "",
        "scenario":    row["scenario"]    if pd.notna(row["scenario"])    else "",
    }, allow_unicode=True)


def call_llm(prompt_template, character_input, taxonomy, log, dry_run=False):
    if dry_run:
        return {"_dry_run": True}

    if taxonomy is not None:
        filled_prompt = prompt_template.format(
            taxonomy=yaml.dump(taxonomy, allow_unicode=True),
            character_input=character_input,
        )
    else:
        filled_prompt = prompt_template.format(character_input=character_input)

    for attempt in range(MAX_RETRIES):
        if log["exhausted"]:
            print("Key exhausted — stopping.")
            return None

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=filled_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            increment_usage(log)

            text = response.text.strip() if response.text else ""
            text = text.replace("```json", "").replace("```", "").strip()

            if not text:
                print(f"  Empty response (attempt {attempt+1}/{MAX_RETRIES}), retrying...")
                time.sleep(SLEEP_BETWEEN_DIMS)
                continue

            return json.loads(text)

        except json.JSONDecodeError:
            print(f"  JSON parse error (attempt {attempt+1}/{MAX_RETRIES}), retrying...")
            time.sleep(SLEEP_BETWEEN_DIMS)

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("  Quota hit — marking exhausted.")
                mark_exhausted(log)
                return None
            else:
                print(f"  API error attempt {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(SLEEP_ON_ERROR)

    print("  All retries failed.")
    return None


def annotate_character(row, prompts, taxonomies, log, dry_run=False):
    char_input = prepare_character_input(row)
    results = {}
    dims = [
        ("demographic",  "demographic",  "demographic"),
        ("occupation",   "occupation",   "occupation"),
        ("space",        "space",        "space"),
        ("relationship", "relationship", "relationship"),
        ("favorability", "favorability", None),
        ("personality",  "personality",  "personality"),
    ]
    for col, prompt_key, tax_key in dims:
        results[col] = call_llm(
            prompts[prompt_key],
            char_input,
            taxonomies.get(tax_key),
            log,
            dry_run,
        )
        if not dry_run:
            time.sleep(SLEEP_BETWEEN_DIMS)
    return results

# ============================================================
# Dry run
# ============================================================

def run_dry_run(df, prompts, taxonomies, log):
    print(f"\n{'='*50}")
    print(f"  DRY RUN — {DRY_RUN_ROWS} rows, no API calls")
    print(f"{'='*50}\n")
    sample = df.sample(n=min(DRY_RUN_ROWS, len(df)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"--- Row {i+1} ---")
        print(f"Bot      : {str(row['bot'])[:40]}")
        print(f"Tags     : {str(row['tags'])[:80]}")
        print(f"Desc     : {str(row['description'])[:80]}")
        result = annotate_character(row, prompts, taxonomies, log, dry_run=True)
        print(f"Mock: {result}\n")
    row        = sample.iloc[0]
    char_input = prepare_character_input(row)
    filled     = prompts["demographic"].format(
        taxonomy=yaml.dump(taxonomies["demographic"], allow_unicode=True),
        character_input=char_input,
    )
    print("--- Sample prompt (demographic, first 600 chars) ---")
    print(filled[:600])
    print("...\nDry run done.")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status",  action="store_true")
    args = parser.parse_args()

    log = load_quota_log()

    if args.status:
        print_quota_status(log)
        return

    prompts    = load_prompts()
    taxonomies = load_taxonomies()
    df         = load_data()
    print_quota_status(log)

    if args.dry_run:
        run_dry_run(df, prompts, taxonomies, log)
        return

    if log["exhausted"]:
        print("Key exhausted for today. Come back tomorrow!")
        return

    unannotated_mask = df[ANNOTATION_COLS].isnull().all(axis=1)
    unannotated_idx  = df[unannotated_mask].index.tolist()
    print(f"Rows to annotate: {len(unannotated_idx)} / {len(df)}\n")

    failed_bots = []

    for count, idx in enumerate(tqdm(unannotated_idx, desc="Annotating")):
        if log["exhausted"]:
            print("\nKey exhausted — stopping.")
            break

        row    = df.loc[idx]
        result = annotate_character(row, prompts, taxonomies, log)

        if all(v is None for v in result.values()):
            failed_bots.append(row["bot"])

        for col in ANNOTATION_COLS:
            df.at[idx, col] = json.dumps(result[col]) if result[col] is not None else None

        time.sleep(SLEEP_BETWEEN_ROWS)

        if (count + 1) % CHECKPOINT_EVERY == 0:
            df.reset_index(drop=True).to_feather(FEATHER_OUTPUT)
            print(f"  [checkpoint] {count+1} rows saved")

    df.reset_index(drop=True).to_feather(FEATHER_OUTPUT)
    print_quota_status(log)
    print(f"Failed bots: {len(failed_bots)}")
    if failed_bots:
        print(failed_bots)


if __name__ == "__main__":
    main()