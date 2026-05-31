import json, pickle, os
import pandas as pd

RAW_JSON = './data/SeSaMe_VersionHistory_Callgraph.v7.json'

# Build lookup: dbid (int) -> version_history_context list
raw = json.load(open(RAW_JSON))
id_to_vh = {}
for record in raw:
    for side in ['first', 'second']:
        obj = record[side]
        dbid = int(obj['dbid'])
        if dbid not in id_to_vh:
            id_to_vh[dbid] = obj['version_history_context']

def build_vh_str(dbid, max_versions):
    """
    Concatenate source code of the N most recent prior versions.
    version_history_context[0] is the current version (always skipped).
    version_history_context[1:max_versions+1] are the prior versions.
    """
    vh = id_to_vh.get(int(dbid), [])
    prior_versions = vh[1:max_versions + 1]   # skip [0], take up to max_versions
    return ''.join(v['commit_source_code'] for v in prior_versions)

# ── Clone detection ──────────────────────────────────────────────────────────
CLONE_IN  = './data/clone_detection'
CLONE_OUT = './data/clone_detection_vh_ablation'
os.makedirs(CLONE_OUT, exist_ok=True)

for split in ['train', 'dev', 'test']:
    df = pd.read_pickle(f'{CLONE_IN}/{split}_blocks.pkl')

    df['t_code_versions_all_1_x'] = df['id1'].apply(lambda x: build_vh_str(x, 1))
    df['t_code_versions_all_3_x'] = df['id1'].apply(lambda x: build_vh_str(x, 3))
    df['t_code_versions_all_1_y'] = df['id2'].apply(lambda x: build_vh_str(x, 1))
    df['t_code_versions_all_3_y'] = df['id2'].apply(lambda x: build_vh_str(x, 3))
    # keep t_code_versions_all_x / _y as-is for vh_all condition

    df.to_pickle(f'{CLONE_OUT}/{split}_blocks.pkl')
    print(f'Saved clone {split}: {df.shape}')

# ── Code classification ──────────────────────────────────────────────────────
CLASS_IN  = './data/classification'
CLASS_OUT = './data/classification_vh_ablation'
os.makedirs(CLASS_OUT, exist_ok=True)

for split in ['train', 'dev', 'test']:
    df = pd.read_pickle(f'{CLASS_IN}/{split}_df.pkl')

    df['t_code_versions_all_1'] = df['id'].apply(lambda x: build_vh_str(x, 1))
    df['t_code_versions_all_3'] = df['id'].apply(lambda x: build_vh_str(x, 3))
    # keep t_code_versions_all as-is for vh_all condition

    df.to_pickle(f'{CLASS_OUT}/{split}_df.pkl')
    print(f'Saved class {split}: {df.shape}')
