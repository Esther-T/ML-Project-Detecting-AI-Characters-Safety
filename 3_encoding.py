"""
Feature Encoding Pipeline: annotated_with/without_nulls.json -> encoded_with/without_nulls.feather
Purpose: Encode LLM annotations and metadata into a binary feature matrix for ML training
Steps:
  1. Load ml_df.feather (bot,y,NSFW,tags,group) and both annotation JSON files
  2. Merge annotations with metadata on bot name
  3. Encode each feature group:
       - demographic  : one-hot (age, gender, race) + binary (victim) + multi-hot (body)
       - occupation   : one-hot category + one-hot sub_category
       - space        : one-hot category + one-hot sub_category
       - relationship : multi-hot category + multi-hot sub_category (multiple entries)
       - personality  : multi-hot category + multi-hot sub_category (multiple entries)
       - favorability : bucket raw score (-1 to 1) -> Like/Neutral/Dislike -> one-hot
       - tags         : multi-hot, filtered to tags appearing in >= 5% of characters
       - NSFW/group   : direct binary (2 cols)
  4. Concatenate all encoded features -> sparse binary feature matrix X
  5. Run side-by-side comparison of with_nulls vs without_nulls (samples, features, ratio)
  6. Save both encoded datasets -> encoded_with/without_nulls.feather
"""

import json
import os
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#Paths
BASE_PATH    = '.'
DATA_PATH    = os.path.join(BASE_PATH, 'data')
ML_FEATHER_PATH = os.path.join(DATA_PATH, 'ml_df.feather') 
ANNOTATION_FILES = {
    'with_nulls':    os.path.join(DATA_PATH, 'annotated_with_nulls.json'),
    'without_nulls': os.path.join(DATA_PATH, 'annotated_without_nulls.json'),
}
OUTPUT_FILES = {
    'with_nulls':    os.path.join(DATA_PATH, 'encoded_with_nulls.feather'),
    'without_nulls': os.path.join(DATA_PATH, 'encoded_without_nulls.feather'),
}

#we only plan to keep tags that appeared in >= 5% of characters
TAG_MIN_FREQ = 0.05 


def encode_demographic(series):
    records = []
    for ann in series:
        ann = ann if isinstance(ann, dict) else {}
        records.append({
            'age':    ann.get('age',    'Unspecified'),
            'gender': ann.get('gender', 'Unspecified'),
            'race':   ann.get('race',   'Unspecified'),
            'victim': ann.get('victim', 'No'),
            'body':   ann.get('body',   []),
        })
    dem_df = pd.DataFrame(records)

    age_dummies    = pd.get_dummies(dem_df['age'],    prefix='age',    dtype=int)
    gender_dummies = pd.get_dummies(dem_df['gender'], prefix='gender', dtype=int)
    race_dummies   = pd.get_dummies(dem_df['race'],   prefix='race',   dtype=int)

    victim_binary = (
        dem_df['victim'].astype(str).str.strip().str.lower() == 'yes'
    ).astype(int).rename('victim')

    body_lists = dem_df['body'].apply(
        lambda x: x if isinstance(x, list) else ([x] if isinstance(x, str) else [])
    )
    mlb_body = MultiLabelBinarizer()
    body_encoded = pd.DataFrame(
        mlb_body.fit_transform(body_lists),
        columns=[f'body_{c}' for c in mlb_body.classes_],
        dtype=int
    )

    return pd.concat(
        [age_dummies, gender_dummies, race_dummies,
         victim_binary.to_frame(), body_encoded],
        axis=1
    )


def encode_single_cat(series, prefix):
    cats, subcats = [], []
    for ann in series:
        ann = ann if isinstance(ann, dict) else {}
        cats.append(ann.get('category',     'Unspecified'))
        subcats.append(ann.get('sub_category', 'Unspecified'))

    cat_dummies    = pd.get_dummies(pd.Series(cats),    prefix=f'{prefix}_cat',    dtype=int)
    subcat_dummies = pd.get_dummies(pd.Series(subcats), prefix=f'{prefix}_subcat', dtype=int)
    return pd.concat([cat_dummies, subcat_dummies], axis=1)


def encode_multi_cat(series, prefix):
    cat_lists, subcat_lists = [], []

    for ann in series:
        if not isinstance(ann, list):
            cat_lists.append([])
            subcat_lists.append([])
            continue

        cats, subcats = [], []
        for entry in ann:
            if not isinstance(entry, dict):
                continue
            cat = entry.get('category', '')
            if isinstance(cat, str) and cat.strip():
                cats.append(cat.strip())
            subcat = entry.get('sub_category', '')
            if isinstance(subcat, list):
                subcats.extend(sc.strip() for sc in subcat
                               if isinstance(sc, str) and sc.strip())
            elif isinstance(subcat, str) and subcat.strip():
                subcats.append(subcat.strip())

        cat_lists.append(cats)
        subcat_lists.append(subcats)

    if all(len(c) == 0 for c in cat_lists):
        cat_lists[0] = ['Unknown']
    if all(len(s) == 0 for s in subcat_lists):
        subcat_lists[0] = ['Unknown']

    mlb_cat = MultiLabelBinarizer()
    cat_encoded = pd.DataFrame(
        mlb_cat.fit_transform(cat_lists),
        columns=[f'{prefix}_cat_{c}' for c in mlb_cat.classes_],
        dtype=int
    )
    mlb_sub = MultiLabelBinarizer()
    sub_encoded = pd.DataFrame(
        mlb_sub.fit_transform(subcat_lists),
        columns=[f'{prefix}_subcat_{c}' for c in mlb_sub.classes_],
        dtype=int
    )
    return pd.concat([cat_encoded, sub_encoded], axis=1)


def encode_favorability(series):
    def bucket(ann):
        if not isinstance(ann, dict):
            return 'Neutral'
        score = ann.get('favorability', 0)
        if score is None:
            return 'Neutral'
        if score > 0:
            return 'Like'
        elif score < 0:
            return 'Dislike'
        return 'Neutral'

    labels = series.apply(bucket)
    return pd.get_dummies(labels, prefix='fav', dtype=int)


def encode_tags(series, min_freq=TAG_MIN_FREQ):
    def parse_tags(t):
        if isinstance(t, list):
            return t
        if isinstance(t, str):
            try:
                parsed = ast.literal_eval(t)
                return parsed if isinstance(parsed, list) else [t]
            except Exception:
                return [tag.strip() for tag in t.split(',')] if ',' in t else [t]
        return []

    tag_lists  = series.apply(parse_tags)
    all_tags   = [tag for tags in tag_lists for tag in tags]
    tag_counts = pd.Series(all_tags).value_counts(normalize=True)
    keep_tags  = set(tag_counts[tag_counts >= min_freq].index)
    print(f"Tags kept after {int(min_freq*100)}% frequency filter: {len(keep_tags)}")

    filtered = tag_lists.apply(lambda tags: [t for t in tags if t in keep_tags])
    mlb = MultiLabelBinarizer()
    encoded = pd.DataFrame(
        mlb.fit_transform(filtered),
        columns=[f'tag_{c}' for c in mlb.classes_],
        dtype=int
    )
    return encoded


#Main encoding function
def encode_dataset(annotation_path, meta_df, output_path, label):
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"{'='*60}")

    with open(annotation_path, 'r') as f:
        all_annotations = json.load(f)
    print(f"Annotations loaded: {len(all_annotations)} characters")

    ann_df = pd.DataFrame.from_dict(all_annotations, orient='index')
    ann_df.index.name = 'bot'
    ann_df = ann_df.reset_index()

    merged = meta_df.merge(ann_df, on='bot', how='inner')
    merged = merged.reset_index(drop=True)
    print(f"Characters after merge: {len(merged)}")

    print("\n  Encoding features...")

    demographic_encoded  = encode_demographic(merged['demographic'])
    print(f"    demographic:  {demographic_encoded.shape[1]} cols")

    occupation_encoded   = encode_single_cat(merged['occupation'],  prefix='occ')
    print(f"    occupation:   {occupation_encoded.shape[1]} cols")

    space_encoded        = encode_single_cat(merged['space'],       prefix='space')
    print(f"    space:        {space_encoded.shape[1]} cols")

    relationship_encoded = encode_multi_cat(merged['relationship'], prefix='rel')
    print(f"    relationship: {relationship_encoded.shape[1]} cols")

    personality_encoded  = encode_multi_cat(merged['personality'],  prefix='pers')
    print(f"    personality:  {personality_encoded.shape[1]} cols")

    favorability_encoded = encode_favorability(merged['favorability'])
    print(f"    favorability: {favorability_encoded.shape[1]} cols")

    tags_encoded         = encode_tags(merged['tags'])
    print(f"    tags:         {tags_encoded.shape[1]} cols")

    nsfw_binary  = merged['NSFW'].astype(int).rename('NSFW')
    raw_group = merged['group']
    if raw_group.dtype == object:
        group_binary = (raw_group.astype(str).str.strip().str.lower() == 'popular').astype(int)
    else:
        group_binary = raw_group.astype(int)
    group_binary = group_binary.rename('group_popular')
    print(f"    NSFW + group: 2 cols")

    X = pd.concat([
        demographic_encoded,
        occupation_encoded,
        space_encoded,
        relationship_encoded,
        personality_encoded,
        favorability_encoded,
        tags_encoded,
        nsfw_binary.to_frame(),
        group_binary.to_frame(),
    ], axis=1)

    y = merged['y']

    #Summary
    print(f"\n  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Samples-to-features ratio: {X.shape[0] / X.shape[1]:.2f}:1")
    sparsity = (X == 0).sum().sum() / X.size * 100
    print(f"  Sparsity: {sparsity:.1f}% zeros (expected >80% for multi-hot)")
    print(f"\n  Target distribution: {y.value_counts().to_dict()}")

    feature_groups = {
        'age':          [c for c in X.columns if c.startswith('age_')],
        'gender':       [c for c in X.columns if c.startswith('gender_')],
        'race':         [c for c in X.columns if c.startswith('race_')],
        'victim':       [c for c in X.columns if c == 'victim'],
        'body':         [c for c in X.columns if c.startswith('body_')],
        'occupation':   [c for c in X.columns if c.startswith('occ_')],
        'space':        [c for c in X.columns if c.startswith('space_')],
        'relationship': [c for c in X.columns if c.startswith('rel_')],
        'personality':  [c for c in X.columns if c.startswith('pers_')],
        'favorability': [c for c in X.columns if c.startswith('fav_')],
        'tags':         [c for c in X.columns if c.startswith('tag_')],
        'NSFW':         [c for c in X.columns if c == 'NSFW'],
        'group':        [c for c in X.columns if c == 'group_popular'],
    }
    print(f"\n  Feature breakdown:")
    for name, cols in feature_groups.items():
        print(f"    {name:15s}: {len(cols):4d}")

    #Save file as .feather
    output_df = pd.concat([merged[['bot']], X, y.rename('y')], axis=1)
    output_df.columns = output_df.columns.astype(str)
    output_df.to_feather(output_path)
    print(f"\n  Saved -> {output_path}")
    print(f"  Reload: df = pd.read_feather('{output_path}')")

    return output_df


if __name__ == '__main__':

    meta_df = pd.read_feather(ML_FEATHER_PATH)[['bot', 'y', 'NSFW', 'tags', 'group']].copy()
    print(f"Meta dataframe loaded: {len(meta_df)} characters")

    results = {}
    for key, ann_path in ANNOTATION_FILES.items():
        results[key] = encode_dataset(
            annotation_path = ann_path,
            meta_df         = meta_df,
            output_path     = OUTPUT_FILES[key],
            label           = key.replace('_', ' ').title(),
        )

    #for comparison
    print(f"\n{'='*60}")
    print("COMPARISON: with_nulls vs without_nulls")
    print(f"{'='*60}")
    for key, df in results.items():
        X = df.drop(['bot', 'y'], axis=1)
        y = df['y']
        print(f"\n  {key}:")
        print(f"    Samples  : {len(X)}")
        print(f"    Features : {X.shape[1]}")
        print(f"    Ratio    : {len(X)/X.shape[1]:.2f}:1")
        print(f"    Target   : {y.value_counts().to_dict()}")
