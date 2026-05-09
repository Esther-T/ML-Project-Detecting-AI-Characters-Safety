"""
LLM Annotation Pipeline: ml_df.feather -> annotated_with_nulls.json
Purpose: Annotate each character across 6 dimensions using Gemini API
Steps:
  1. Load ml_df.feather and taxonomies (YAML) + prompts (TXT) from disk
  2. Compute per-character unsafety score -> bot_df, normalize by platform mean, assign y labels -> ml_df
  3. Annotate each character via Gemini API across 6 dimensions:
     demographic, occupation, space, relationship, favorability, personality
  4. Checkpoint results incrementally to JSON; skip already-annotated characters on resume
  5. Drop complete failures (all 6 fields null) and fill partial nulls with field defaults
  6. Save cleaned annotations -> annotated_with_nulls.json
Usage:
    1. pip install pandas numpy pyyaml pyarrow python-dotenv google-genai tqdm
    2. Create a .env file in the same directory as this script with:
       GEMINI_API_KEYS=key1,key2,key3,...
"""

import os
import json
import time
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

#Paths
BASE_PATH      = os.path.expanduser("~/ml_project")         
DATA_PATH      = os.path.join(BASE_PATH, "data")
TAXONOMY_PATH  = os.path.join(BASE_PATH, "taxonomies")
PROMPT_PATH    = os.path.join(BASE_PATH, "prompts")

CHECKPOINT_PATH       = os.path.join(DATA_PATH, "annotated_checkpoint.json") 
CLEANED_WITH_NULLS    = os.path.join(DATA_PATH, "annotated_with_nulls.json") 

#Load Dataset
file_path = os.path.join(DATA_PATH, "ml_df_part1.feather") #replace with ml_df.part2.feather
df = pd.read_feather(file_path)
print("Dataset loaded:")
print(df.info())

#Load Taxonomies
yaml_files = {}
for filename in sorted(os.listdir(TAXONOMY_PATH)):
    if filename.endswith(".yaml"):
        filepath = os.path.join(TAXONOMY_PATH, filename)
        with open(filepath, "r") as f:
            yaml_files[filename] = yaml.safe_load(f)
        print(f"Loaded taxonomy: {filename}")

demographic_taxonomy  = yaml_files["VII_Demographic.yaml"]
occupation_taxonomy   = yaml_files["VIII_Occupation.yaml"]
space_taxonomy        = yaml_files["IX_Space.yaml"]
relationship_taxonomy = yaml_files["X_Relationship.yaml"]
personality_taxonomy  = yaml_files["XI_Personality.yaml"]

#Load Prompts
prompt_files = {}
for filename in sorted(os.listdir(PROMPT_PATH)):
    if filename.endswith(".txt"):
        filepath = os.path.join(PROMPT_PATH, filename)
        with open(filepath, "r") as f:
            prompt_files[filename] = f.read()
        print(f"Loaded prompt: {filename}")

demographic_prompt  = prompt_files["demographic_prompt.txt"]
occupation_prompt   = prompt_files["occupation_prompt.txt"]
space_prompt        = prompt_files["space_prompt.txt"]
relationship_prompt = prompt_files["relationship_prompt.txt"]
favorability_prompt = prompt_files["favorability_prompt.txt"]
personality_prompt  = prompt_files["personality_prompt.txt"]

print("\nAll external data loaded successfully!")

#Preprocess
df["is_unsafe"] = df["judge_score"] >= 3

bot_df = df.groupby("bot").agg(
    raw_unsafety_score=("is_unsafe", "mean"),
    platform=("platform", "first"),
    group=("group", "first"),
    NSFW=("NSFW", "first"),
    description=("description", "first"),
    tags=("tags", "first"),
    scenario=("scenario", "first"),
).reset_index()

platform_means = bot_df.groupby("platform")["raw_unsafety_score"].mean().reset_index()
platform_means.rename(columns={"raw_unsafety_score": "platform_mean"}, inplace=True)
bot_df = bot_df.merge(platform_means, on="platform", how="left")
bot_df["normalized_score"] = bot_df["raw_unsafety_score"] - bot_df["platform_mean"]

mean_score       = bot_df["normalized_score"].mean()
std_score        = bot_df["normalized_score"].std()
unsafer_threshold = mean_score + std_score
safer_threshold   = mean_score

def assign_y_label(score):
    if score >= unsafer_threshold:
        return 1
    elif score < safer_threshold:
        return 0
    return np.nan

bot_df["y"] = bot_df["normalized_score"].apply(assign_y_label)
ml_df = bot_df.dropna(subset=["y"]).copy()

print(f"Total Characters: {len(bot_df)}")
print(f"Characters kept for ML training: {len(ml_df)}")
print("\nDistribution of y:")
print(ml_df["y"].value_counts())

#Load API key
load_dotenv() 

api_keys_env = os.getenv("GEMINI_API_KEYS", "")
raw_keys = [k.strip() for k in api_keys_env.split(",") if k.strip()]

if not raw_keys:
    raise ValueError(
        "No API keys found. Create a .env file with:\n"
        "  GEMINI_API_KEYS=your_key1,your_key2,..."
    )

MODEL_NAME = "gemini-3.1-flash-lite"

API_KEYS = [{"key": k, "model": MODEL_NAME} for k in raw_keys]
key_index = 0

print(f"\nLoaded {len(API_KEYS)} API key(s), model: {MODEL_NAME}")

#Annotation Helpers
def prepare_character_input(row):
    character = {
        "tags": row["tags"],
        "description": row["description"] if pd.notna(row["description"]) else "",
        "scenario":    row["scenario"]    if pd.notna(row["scenario"])    else "",
    }
    return yaml.dump(character, allow_unicode=True)


def call_llm(prompt_template, character_input, taxonomy=None, max_retries=3):
    global key_index

    if taxonomy is not None:
        filled_prompt = prompt_template.format(
            taxonomy=yaml.dump(taxonomy, allow_unicode=True),
            character_input=character_input,
        )
    else:
        filled_prompt = prompt_template.format(character_input=character_input)

    for attempt in range(max_retries):
        for _ in range(len(API_KEYS)):
            try:
                current_entry = API_KEYS[key_index]
                client = genai.Client(api_key=current_entry["key"])

                response = client.models.generate_content(
                    model=current_entry["model"],
                    contents=filled_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )

                text = response.text.strip() if response.text else ""
                text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)

            except json.JSONDecodeError:
                print(f"  JSON parse error (attempt {attempt + 1}), retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"  API error on key {key_index} (attempt {attempt + 1}): {e}")
                key_index = (key_index + 1) % len(API_KEYS)
                time.sleep(2)

    print("  All retries failed, returning None")
    return None


def annotate_character(row):
    character_input = prepare_character_input(row)
    return {
        "demographic":  call_llm(demographic_prompt,  character_input, demographic_taxonomy),
        "occupation":   call_llm(occupation_prompt,   character_input, occupation_taxonomy),
        "space":        call_llm(space_prompt,         character_input, space_taxonomy),
        "relationship": call_llm(relationship_prompt, character_input, relationship_taxonomy),
        "favorability": call_llm(favorability_prompt, character_input, taxonomy=None),
        "personality":  call_llm(personality_prompt,  character_input, personality_taxonomy),
    }

#8. Run annotations with checkpointing
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r") as f:
        all_annotations = json.load(f)
    print(f"Resuming from checkpoint: {len(all_annotations)} already annotated")
else:
    all_annotations = {}
    print("Starting fresh annotation")

already_done = set(all_annotations.keys())
remaining = ml_df[~ml_df["bot"].isin(already_done)].reset_index(drop=True)
print(f"Remaining to annotate: {len(remaining)} characters")

BATCH_SIZE  = 1030
failed_bots = []
batch = remaining.head(BATCH_SIZE)
print(f"Running batch of {len(batch)} characters")

for idx, row in tqdm(batch.iterrows(), total=len(batch), desc="Annotating"):
    bot_name = row["bot"]
    result   = annotate_character(row)

    if all(v is None for v in result.values()):
        failed_bots.append(bot_name)
        continue

    all_annotations[bot_name] = result

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(all_annotations, f)

    time.sleep(4)

print(f"\nBatch complete!")
print(f"Total annotated so far: {len(all_annotations)}")
print(f"Failed in this batch:   {len(failed_bots)}")
print(f"Remaining after batch:  {len(ml_df) - len(all_annotations)}")

#Clean the annotations
ALL_FIELDS = ["demographic", "occupation", "space", "relationship", "favorability", "personality"]

with open(CHECKPOINT_PATH, "r") as f:
    raw_annotations = json.load(f)
print(f"\nLoaded {len(raw_annotations)} annotated characters for cleaning")

def is_complete_failure(ann):
    return all(ann.get(field) is None for field in ALL_FIELDS)

complete_failures = [bot for bot, ann in raw_annotations.items() if is_complete_failure(ann)]
print(f"Complete failures (all 6 null): {len(complete_failures)}")

FIELD_DEFAULTS = {
    "demographic":  {"age": "Unknown", "body": ["Unknown"], "gender": "Unknown",
                     "race": "Unknown", "victim": "Unknown"},
    "occupation":   {"category": "None", "sub_category": "None"},
    "space":        {"category": "None", "sub_category": "None"},
    "relationship": [{"category": "None", "sub_category": "None"}],
    "personality":  [{"category": "None", "polarity": "neutral", "sub_category": "None"}],
    "favorability": {"favorability": None},
}

def fill_nulls(ann):
    filled = ann.copy()
    for field, default in FIELD_DEFAULTS.items():
        if filled.get(field) is None:
            filled[field] = default
    return filled

cleaned_with_nulls = {
    bot: fill_nulls(ann)
    for bot, ann in raw_annotations.items()
    if bot not in complete_failures
}

with open(CLEANED_WITH_NULLS, "w") as f:
    json.dump(cleaned_with_nulls, f, indent=2)

print(f"\nOriginal:                   {len(raw_annotations)} characters")
print(f"Complete failures dropped:  {len(complete_failures)} characters")
print(f"Final (nulls filled):       {len(cleaned_with_nulls)} characters")
print(f"\nOutput written to: {CLEANED_WITH_NULLS}")
