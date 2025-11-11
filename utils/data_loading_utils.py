import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys, os
import math
import azure_ml_configs
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def create_acute_readmission_data():
    """
    This function shows how we created a dataset of clinical notes for acute readmission prediction.
    A pandas dataframe of clinical notes (df) are matched to dataframes where cases and controls, for acute readmission prediction, have already been defined during another project investigating the same outcome.
    Cases are encounters (hospitalizations in a psychiatric unit) of patients whose next (re)admission happens within 30 days after discharge.
    Please read the paper for more details.

    The matched data was preprocessed post-hoc (preprocessing/name_removal_scripts), but if needing to update the matches, it might be better to use data that has already been preproccesed.
    """

    # Initiate workspace
    workspace_id =  azure_ml_configs.workspace_id
    subscription_id = azure_ml_configs.subscription_id
    resource_group = azure_ml_configs.resource_group
    workspace_name = azure_ml_configs.workspace_name

    data_location = "../data/acuteReadmission/"

    # Get a handle to the workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    data_asset = ml_client.data.get(name="clinicalNote44M", version=1) 
    print(f"Data asset URI: {data_asset.path}")

    df = pd.read_csv(data_asset.path)

    cases = pd.read_parquet(data_location + 'acute-copy.parquet')

    controls = pd.read_parquet(data_location + 'discharges_psy_24h-copy.parquet')

    controls['Acute'] = 0
    cases['Acute'] = 1

    events = pd.concat([cases, controls])
    events = events.reset_index(drop = True) # VERY IMPORTANT TO RUN TO ENSURE NO CASE AND CONTROL HAVE THE SAME INDEX!

    events.to_parquet(data_location+"events_acute_labels.parquet")

    # Make sure patients exist in both dataframes
    df["in_events"] = df.PatientDurableKey.isin(events.PatientDurableKey)
    dfnew = df[df.in_events==True].copy()
    print("df",len(df))
    print("dfnew, patients in events",len(dfnew))

    print("events", len(events))
    events["in_notesdf"] = events.PatientDurableKey.isin(dfnew.PatientDurableKey)
    events = events[events.in_notesdf==True].copy()
    print("events, patients in dfnew",len(events))

    # make new encounter column, as acute and non acute have different encounter columns in events dataframe
    events["encounter"] = [int(i) if math.isnan(i)==False else int(j) for (i,j) in list(zip(events.EncounterKey_dis.values, events.EncounterKey.values))]

    # make sure the encounters exist in both dataframes
    events["encounter_in_notesdf"] = events.encounter.isin(dfnew.EncounterKey)
    events = events[events.encounter_in_notesdf==True].copy()
    print("events, encounters in dfnew",len(events))

    dfnew["encounter_in_events"] = dfnew.EncounterKey.isin(events.encounter)
    dfnew = dfnew[dfnew.encounter_in_events==True].copy()
    print("dfnew, encounters in events", len(dfnew))

    # Match labels!
    dfnew["Acute"] = dfnew.EncounterKey.isin(events[events.Acute==1].encounter)
    dfnew["Acute"] = dfnew["Acute"].astype(int)


    dfnew = train_val_test_split(dfnew)

    dfnew.to_csv("../data/acuteReadmission/clinicalNote_AcuteReadmissions_210823.csv")

    return dfnew



def update_acute_readmission_data(notes_path="../data/acuteReadmission/clinicalNote_AcuteReadmissions_NamesRemoved_180923.csv"):
    """ 
    Function to update acute readmission data when logic for defining acute readmissions changed such that encounters should be re-matched.
    The new logic was improved to not mistake some movements back and forth between psychiatric and somatic units, during admissions, as new (re)admissions.
    
    """

    df = pd.read_csv(notes_path)

    data_location= "../data/acuteReadmission/" 

    ## Define cases and controls
    # Load tables
    cases = pd.read_parquet(data_location + 'acute_after_logic-copy.parquet')
    controls = pd.read_parquet(data_location + 'controls-copy.parquet')

    controls['Acute'] = 0
    cases['Acute'] = 1

    events = pd.concat([cases, controls])
    events = events.reset_index(drop = True) # VERY IMPORTANT TO RUN TO ENSURE NO CASE AND CONTROL HAVE THE SAME INDEX!

    # Make sure patients exist in both dataframes
    df["in_events"] = df.PatientDurableKey.isin(events.PatientDurableKey)
    dfnew = df[df.in_events==True].copy()
    print("df",len(df))
    print("dfnew, patients in events",len(dfnew))

    print("events", len(events))
    events["in_notesdf"] = events.PatientDurableKey.isin(dfnew.PatientDurableKey)
    events = events[events.in_notesdf==True].copy()
    print("events, patients in dfnew",len(events))
    
    # make new encounter column, as acute and non acute have different encounter columns in events dataframe
    # if EncounterKey_dis_last exists, take that (the last encounter in several consecutive encounters for one admission, if the patient was moved etc)
    # else if EncounterKey_dis_first exists, take that 
    # else if above fails, the instance is a control and there is just one regular EncounterKey
    e_keys = []
    for i,j,k in list(zip(events.EncounterKey_dis_last.values,events.EncounterKey_dis_first.values, events.EncounterKey.values)):
        if math.isnan(i)==False:
            e_keys.append(int(i))
        elif math.isnan(j)==False:
            e_keys.append(int(j))
        else:
            e_keys.append(int(k))

    events["encounter"] = e_keys

    # make sure the encounters exist in both dataframes
    events["encounter_in_notesdf"] = events.encounter.isin(dfnew.EncounterKey)
    events = events[events.encounter_in_notesdf==True].copy()
    print("events, encounters in dfnew",len(events))

    dfnew["encounter_in_events"] = dfnew.EncounterKey.isin(events.encounter)
    dfnew = dfnew[dfnew.encounter_in_events==True].copy()
    print("dfnew, encounters in events", len(dfnew))

    # Match labels!
    dfnew["Acute"] = dfnew.EncounterKey.isin(events[events.Acute==1].encounter)
    dfnew["Acute"] = dfnew["Acute"].astype(int)

    dfnew.drop(columns=["Unnamed: 0", "in_events", "encounter_in_events"], inplace=True)
    dfnew.to_csv("../data/acuteReadmission/clinicalNote_AcuteReadmissions_NamesRemoved_161023.csv")
    
    return dfnew



def train_val_test_split(df):
    """Makes train, val and test splits of data such that no patient appears in more than one split.

    Parameters
    ----------
    df : pandas dataframe (clinical notes) containing, at least, the columns "PatientDurableKey", "EncounterKey", "CreationInstant".

    Returns
    -------
    Object
        df sorted by patient, encounter and note creation date with a new column names "set", identifying which data split ("train" 70%, "val" 10%, "test" 20%) each instance belongs to.
    
    """

    # sort the dataframe by patient id, encounter and date
    df.sort_values(by=["PatientDurableKey", "EncounterKey", "CreationInstant"],inplace=True)
    
    # get unique patient ids
    samplelist = df["PatientDurableKey"].unique()

    # make train, val and test samples based on shuffled patient ids
    train_val, test = train_test_split(samplelist, test_size=0.2, random_state=5, shuffle=True)
    train, val = train_test_split(train_val, test_size=0.13,random_state=5)

    # check num patients in each and the percentage size of each set
    print(len(train), len(val), len(test))
    print(len(train)/len(samplelist), len(val)/len(samplelist), len(test)/len(samplelist))
    # 70%, 10%, 20%

    def mapping(x):
        if x in train:
            return "train"
        elif x in val:
            return "val"
        else:
            return "test"

    # make new column showing which set a patient belongs to
    df["set"] = df["PatientDurableKey"].apply(lambda x: mapping(x))
    
    return df