# PsyRoBERTa

This is the code repository for the paper (preprint) **Evaluating large language models for predicting psychiatric acute readmissions from clinical notes of population-based EHR**.

![plot](./study_overview.jpg)

## Abstract
Psychiatric patients often have complex symptoms and anamneses recorded as unstructured clinical notes. Large language models (LLM) now enable large-scale utilization of text data; however, there is a current lack of LLMs specialized for psychiatric clinical data, as well as non-English data, haltering the application of LLMs across diverse clinical domains and countries. We present PsyRoBERTa: the first LLM specialized for clinical psychiatry, using population-based data with the currently largest collection of clinical notes of psychiatric relevancy (~44 million notes) covering the eastern half of Denmark. The model was evaluated against three publicly available models, pretrained on either public general- or medical-domain text, and a baseline logistic regression classifier. Through extensive evaluations, we investigated the effect of domain-specific pretraining on predicting acute readmissions in psychiatric hospitals, explored important features, and reflected on (dis)advantages of LLMs. PsyRoBERTa succeeded in outperforming prior models (AUC=0.74), capturing information aligning with clinical practice, and additionally recognizing psychiatric diagnoses (AUC=0.85). This demonstrates the importance of domain-pretraining and the potential of LLMs to leverage psychiatric clinical notes for enhancing prediction of psychiatric outcomes.

## Where is the data?
The data behind this repository and paper is **not** publicly available due to its sensitive content, but it is accessible for researchers with appropriate legal permissions on secured servers.

On the server, the data files needed for this study are:

| File name| Location  | Columns used | Description  
| :---------------- | :------:  | :---- | :---- |
|large_collected_namelist_edited.txt  | preprocessing/name_removal_scripts |  | Curated list of Danish and international names. |
| clinicalNote44M.csv | Datastore "sp_data" | Text, Type, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M". This is the raw (unprocessed) data file of clinical notes. |
|clinicalNote44M_NamesRemoved.csv | Datastore "researcher_data" | text_names_removed_step2, Type, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M_NamesRemoved". This is the processed version of the above file after running name removal scripts. |
| clinicalNote44M_NamesRemoved_<br>MLM_train_randompart1.csv | Datastore "researcher_data" | text_names_removed_step2, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M_pretraindata_randompart1" |
| clinicalNote44M_NamesRemoved_<br>MLM_train_randompart2.csv | Datastore "researcher_data" | text_names_removed_step2, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M_pretraindata_randompart2" |
| clinicalNote44M_NamesRemoved_<br>MLM_train_randompart3.csv | Datastore "researcher_data" | text_names_removed_step2, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M_pretraindata_randompart3" |
| clinicalNote44M_NamesRemoved_<br>MLM_train_randompart4.csv | Datastore "researcher_data" | text_names_removed_step2, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote44M_pretraindata_randompart4" |
| clinicalNote_AcuteReadmissions_<br>NamesRemoved_150124_1343.csv | Datastore "researcher_data" | text_names_removed_step2, Acute, set, Type, PatientDurableKey, EncounterKey, CreationInstant | Used as data asset "clinicalNote_AcuteReadmission" |
| clinicalNote_AcuteReadmissions_<br>NamesRemoved_DeDup_190324.csv | Datastore "researcher_data/terne" | DedupCont, Acute, set, Type, PatientDurableKey, EncounterKey, CreationInstant  | Used as data asset "clinicalNote_AcuteReadmission_DedupCont". This is the de-duplicated version of the above data asset "clinicalNote_AcuteReadmission" |
| acute_after_logic-copy.parquet | data/acuteReadmission | PatientDurableKey, EncounterKey, EncounterKey_dis_last, EncounterKey_dis_first | File containing cases for acute readmission predcition. | 
| controls-copy.parquet | data/acuteReadmission | PatientDurableKey, EncounterKey, EncounterKey_dis_last, EncounterKey_dis_first |  File containing controls for acute readmission predcition. |
| events_acute_labels.parquet | data/acuteReadmission/ | PatientDurableKey, Date_dis | File used for calculating patients' age at time of discharge. |
| afregningsdiagnose-copy.parquet | data/acuteReadmission/ | PatientDurableKey, EncounterKey, SKSCode, IsActionDiagnosis | File with patient diagnoses.|
|patients.parquet  | Datastore "sp_data" | DurableKey, BirthDate | Patients meta data file. |
| encounters.parquet | Datastore "sp_data" | EncounterKey, PatientDurableKey, DepartmentKey | Encounters meta data file. |
| departments.parquet | Datastore "sp_data" | EncounterKey, PatientDurableKey, DepartmentKey, RegionId | Hospital departments meta data file.|
|  |  |  |  |


## How do I run the code?
Since the data is hosted on private, secure Microsoft Azure servers, it is not possible to run code as is in this repository. To reproduce our results, you need specific data access permissions. The main purpose of sharing our code is transparency in the development of PsyRoBERTa and how the analyses were carried out. You can of course run the code with your own data with a bit a tweaks. Below, we explain what parts makes the code "un-runable" without access to the server, which hopefully makes it clear what you need to edit to run (parts of) the code with your own data, as well as how to run it on the server if you do have access.

## Code specific for Azure ML Studio

!NB The Azure ML code repository might be moved to databricks in the near future. The below code and descriptions hereof might need to be adapted for the new system to run the code there.

### Data Stores and Data Assets

We load data either from the Sundhedsplatform (SP) "data store", where raw data deliveries are stored, or from "data assets" that we have created after processing the raw data. Data assets points to files located in in a data store folder for researcher (processed) data, and these assets are more easy to work with when running "jobs" in Azure ML Studio.

For instance, when reading data from the datastore, the syntax used is:
```python
datastore_name = 'sp_data'
datastore = Datastore.get(workspace, datastore_name)

datastore_paths = [(datastore, '/patients.parquet')] 
ds = Dataset.Tabular.from_parquet_files(path=datastore_paths)
dfP = ds.to_pandas_dataframe()
```
When reading a data asset, the syntax used is:

```python
data_asset = ml_client.data.get(name="clinicalNote_AcuteReadmission", version=1) 
print(f"Data asset URI: {data_asset.path}")
df = pd.read_csv(data_asset.path)
```
In both cases, we eventually read the data as ```pandas``` dataframes.

#### To run the code with your own data, outside of Azure ML Studio... 
... you can delete the Azure data loading code snippets and insert your own data paths to load as ```pandas``` dataframes.

In the folders pretraining/ and finetuning/, there are empty folders where pretrained models should be placed (roberta-base-danish, danish-bert-botxo, MeDa-Bert and psyroberta_part4). Download the language model you want to use and insert the files in the respective folder. PsyRoBERTa can only be accessed with permission on the secure server. (On the server, these folders, as well as the folder result_files/, are not empty.)

## Citations
If you found our code useful, please cite our paper (currently a preprint and under review)! :)

```bibtex
@article{Thorn Jakobsen2025.03.07.25323558,
    author = {Thorn Jakobsen, Terne Sasha and Crist{\`o}bal C{\'o}ppulo, Enric and Rasmussen, Simon and Benros, Michael Eriksen},
    title = {Evaluating large language models for predicting psychiatric acute readmissions from clinical notes of population-based EHR},
    year = {2025},
    doi = {10.1101/2025.03.07.25323558},
    publisher = {Cold Spring Harbor Laboratory Press},
    URL = {https://www.medrxiv.org/content/early/2025/11/01/2025.03.07.25323558},
    journal = {medRxiv}
}
```
