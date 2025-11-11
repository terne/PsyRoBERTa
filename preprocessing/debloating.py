import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from tqdm import tqdm
from azure.ai.ml import MLClient#, Input, command
from azure.identity import DefaultAzureCredential
import sys
sys.path.append("..")
from utils import azure_ml_configs


# Get a handle to the workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=azure_ml_configs.subscription_id,
    resource_group_name=azure_ml_configs.resource_group,
    workspace_name=azure_ml_configs.workspace_name,
)


def DedupCont(df, text_column_name="text_names_removed_step2"):

    """
    Code based on the DedupCont method of Liu et al. (2022) and adapted from https://github.com/JHLiu7/EHR-deduplication/blob/main/deduper.py
    Liu, J., Capurro, D., Nguyen, A.N., & Verspoor, K.M. (2022). "Note Bloat" impacts deep learning-based NLP models for clinical prediction tasks. Journal of biomedical informatics, 104149.
    
    
    """
    
    # make sure the dataframe is sorted by encounter and date
    df.sort_values(by=["PatientDurableKey","EncounterKey","CreationInstant"], inplace=True)
    
    # make list of notes splitted into sections
    # the data has no newlines but it has many instances of consecutive spaces that seem to be where a newline would have occured
    # The list comprehension therefore does the following:
    # - replace annying characters • and \xa0 appropriately
    # - when consecutive spaces begin with ":" they are often test result and we do not want a newline, therefore we replace those spaces with a single space and keep colon
    # - then, consuctive spaces of two or more are replaced by a newline
    # - finally, the instance is split on the newlines
    print("Preparing text...")
    text_series_notes = [re.sub("\s\s\s*","\n",re.sub(":\s\s\s*",": ",i.replace("•","").replace("\xa0", " "))).strip().split("\n") for i in tqdm(df[text_column_name])]
    text_series_encounters = df.EncounterKey.values
    
    lines_before = sum([len(i) for i in text_series_notes])
    print("Lines before deduplication:",lines_before)
    
    new_text_series = []

    prev_enc = text_series_encounters[0]
    note = text_series_notes[0]

    # get first text for comparison
    state_text = [t.strip() for t in text_series_notes[0]]
    new_text_series.append(state_text)

    print("Deduplication...")
    for j in tqdm(range(1,len(text_series_notes))):

        state_text = list(OrderedDict.fromkeys(state_text))

        next_text = [t.strip() for t in text_series_notes[j]]
        next_text = list(OrderedDict.fromkeys(next_text))

        # only remove duplication within enounters
        # when seeing a new encounter, start over with fresh state
        if text_series_encounters[j] != prev_enc:
            state_text = []
            state_text.extend(next_text)
            new_text_series.append(state_text)
            prev_enc = text_series_encounters[j]
        else:
            # compare
            new_piece = [l for l in next_text if l not in state_text]

            # append and update
            new_text_series.append(new_piece)
            state_text.extend(new_piece)
            prev_enc = text_series_encounters[j]
    
    lines_after = sum([len(i) for i in new_text_series])
    print("Lines after deduplication:", lines_after)
    print("Difference:", lines_before-lines_after)
    
    print("Collecting final result...")
    final_text_series, to_keep = [], []
    for i, text in enumerate(tqdm(new_text_series)):
        if len(text) > 0:
            text_ = ' '.join(text).strip()
            if len(text_) > 0:
                # non-empty note
                final_text_series.append(text_)
                to_keep.append(i)
            else:
                continue
        else:
            continue
    to_keep = np.array(to_keep)
    
    num_empty_notes = len(df)-len(to_keep)
    print("Empty notes after deduplication (removed):", num_empty_notes)
    print("Non-empty notes:", len(to_keep))
    #make a new dataframe
    df_dedup = df.iloc[to_keep].copy()
    print("Length of dataframe before and after:", len(df), "vs", len(df_dedup))
    print("Patients before and after: ", len(df.PatientDurableKey.unique()), "vs", len(df_dedup.PatientDurableKey.unique()))
    print("Encounters before and after: ", len(df.EncounterKey.unique()), "vs", len(df_dedup.EncounterKey.unique()))
    assert len(df_dedup)==len(to_keep)==len(final_text_series)
    df_dedup["DedupCont"] = final_text_series
    return df_dedup

data_asset = ml_client.data.get(name="clinicalNote_AcuteReadmission", version=1) 
print(f"Data asset URI: {data_asset.path}")

df = pd.read_csv(data_asset.path)
df.drop(columns=["Unnamed: 0"], inplace=True)

new_df = DedupCont(df)
new_df.reset_index(inplace=True, drop=True)
new_df.to_csv("clinicalNote_AcuteReadmissions_NamesRemoved_DeDup_190324.csv")