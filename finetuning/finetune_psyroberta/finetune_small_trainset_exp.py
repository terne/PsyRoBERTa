from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from tokenizers import processors
from sklearn.utils import class_weight
import pandas as pd
import os
import sys
from itertools import chain
#sys.path.append("../../") # append parent to access src
#from src.utils.data_loader import ClinicalNotesDataset
#from src.utils.data_loading_utils import acute_readmission_data, toydata
from trainer import seq_classification_trainer
import argparse
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    
    # directories
    parser.add_argument("--data", type=str, default=None)
    #parser.add_argument("--tokens_path", type=str, default="../../data/acuteReadmission/tokenized_data/")
    parser.add_argument("--pretrained_model_path", type=str, default="../models/")
    parser.add_argument("--model_name", type=str, default="danish-bert-botxo")
    # checkpoint_dir should be set in shell script and include seed and model name, e.g. "../../output/checkpoints/roberta-base-danish/42"
    parser.add_argument("--checkpoint_dir", type=str, default="../../output/checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")

    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    #parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, besides at every epoch.",
    )
    parser.add_argument(
        "--add_special_tokens_notes",
        action="store_true",
        help="If passed, will add preproccessing/de-identification masks as speciel tokens.",
    )

    # Optimizer
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01, #0.0
                        help="Weight decay for AdamW")
    parser.add_argument("--max_gradient_norm",
                        type=float,
                        default=10.0,
                        help="Max. norm for gradient norm clipping")
    # Scheduler
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="mlflow",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, `"clearml"` and  `"mlflow"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )                     
        
    # Experiment parameters
    parser.add_argument("--nrows", type=int, default=None, help="To load only n rows from data for development tests.")
    parser.add_argument("--text_column_name", type=str, default="text_names_removed_step2")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--discharge_notes_only",
                        action="store_true",
                        help="If passed, filters data to only include discharge summaries.")
    parser.add_argument("--max_seq_splits", 
                        type=int, 
                        default=None, 
                        help="The number of splits that long notes will be split to. Default None means that every part of a long note is included. However, some notes are extremely long (tens of thousands tokens) compared to the majority (median ~130 tokens). Based on descriptive analysis of sequence lengths across regions and psychiatric centers, max_seq_splits=4 might be appropriate, because for the center with longest notes, 75% of are less than 1600 tokens long.")
    parser.add_argument("--scale_loss", action="store_true")
    parser.add_argument("--do_not_concat", action="store_true", help="By default, notes are concatenated on encounters with a seperator token. If this argument is passed, notes will not be concatenated.")
    parser.add_argument("--frac", type=float, default=0.1, help="Size (fraction) of train set to finetune with in experiments with varying train set sizes.")


    logger = get_logger(__name__)
    args = parser.parse_args()
    data_path = args.data
    model_path = args.pretrained_model_path
    model_name = args.model_name
    checkpoint_dir = args.checkpoint_dir # with azure, should be within "./outputs"
    print(f'Checkpoints will be saved in {checkpoint_dir}')
    # creat checkpoint dir if it does not exist
    #if not os.path.exists(checkpoint_dir):
    #    os.makedirs(checkpoint_dir)

    batch_size = args.batch_size
  
    #print("highrisk=",args.highrisk)
    print("max_seq_splits=", args.max_seq_splits)
    print("disharge notes only =", args.discharge_notes_only)

    # setting random seeds
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # increase timeout
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600000))

    # randomness for DataLoader, see https://pytorch.org/docs/stable/notes/randomness.html
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.random_seed)

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.log_dir #args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path+model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_name, local_files_only=True)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    
    tokenizer._tokenizer.post_processor = processors.RobertaProcessing(
    sep=("</s>", tokenizer._tokenizer.token_to_id("</s>")),
    cls=("<s>", tokenizer._tokenizer.token_to_id("<s>"))
    )

    # add special tokens
    if args.add_special_tokens_notes:
        special_tokens_dict = {'additional_special_tokens': ['[Name]','[Url]','[Email]','[CprNumber]', '[Phone]', '[Address]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # We resize the embeddings only to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        #On A100, choosing multiples of larger powers of two up to 128 bytes (64 for FP16) can further improve efficiency. 
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    
    # loading and prepraring data
    cols = [args.text_column_name, "Acute", "set", "Type", "PatientDurableKey", "EncounterKey", "CreationInstant"]
    df = pd.read_csv(data_path, usecols=cols, nrows=args.nrows)
    # make sure the data is sorted by patient id, encounter and date
    df.sort_values(by=["PatientDurableKey", "EncounterKey", "CreationInstant"],inplace=True)
    #rename main columns of interest
    df.rename(columns={args.text_column_name: "text", "Acute": "label"}, inplace=True)

    if args.discharge_notes_only:
        df = df[df["Type"].str.contains("Udskrivningsresume|Udskrivningsresumé")==True].copy()
        # to do: take 1 when there are more than 1.

    
    concatenate_notes = True
    if args.do_not_concat:
        concatenate_notes = False
    
    # concatenating texts on patient and encounter id
    if concatenate_notes:
        df = df.groupby(["PatientDurableKey", "EncounterKey", "label", "set"]).text.apply(f'{tokenizer.sep_token}'.join).reset_index()

    train_set = df[df.set=="train"].sample(frac=args.frac, random_state=42)
    print("train sample size:", args.frac)
    print("train sample:",len(train_set))
    print("train sample label dist:",train_set.label.value_counts())
    print("train sample encounters:",train_set.EncounterKey.unique())
    

    # train and val sets
    data_dict = {
        "train": Dataset.from_pandas(train_set),
        "validation": Dataset.from_pandas(df[df.set=="val"]),
        "test": Dataset.from_pandas(df[df.set=="test"])
    }
    raw_datasets = DatasetDict(data_dict)

    text_column_name = "text"
    
    def tokenize_function(examples):
        input_ids = []
        attention_masks = []
        labs = []
        patientids = []
        encounterids = []
        for x,y, patient_id, encounter_id in list(zip(examples["text"], examples["label"], examples["PatientDurableKey"], examples["EncounterKey"])):
            encoded_dict = tokenizer.encode_plus(
                x,  # Sentence to encode
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]' or equivelant for roberta
                max_length=512,  # Pad & truncate all sentences.
                padding="max_length", #(needing to specify truncation=True depends on version)
                truncation=True,
                return_overflowing_tokens=True, # return lists of tokens above 512 
                return_offsets_mapping=True,
                stride=32, # The stride used when the context is too large and is split across several features.
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt'  # Return pytorch tensors.
            )
            for inputs, attentions in list(zip(encoded_dict['input_ids'],encoded_dict['attention_mask']))[:args.max_seq_splits]:
                #print(i.shape)
                # Add the encoded sentence to the list.
                input_ids.append(inputs)
                #And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(attentions)
                labs.append(y)
                patientids.append(patient_id)
                encounterids.append(encounter_id)
        assert len(input_ids) == len(attention_masks) == len(labs) == len(patientids) == len(encounterids)
        sample = {"inputs": input_ids,
                "attn_masks": attention_masks,
                "labels": labs,
                "patient_id": patientids,
                "encounter_id": encounterids}
        return sample
    
    
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=raw_datasets['train'].column_names,
            #load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
    
    
    tokenized_datasets["train"].set_format(type='pt')
    tokenized_datasets["validation"].set_format(type='pt')
    tokenized_datasets["test"].set_format(type='pt')

    traindata = tokenized_datasets["train"]
    valdata = tokenized_datasets["validation"]
    testdata = tokenized_datasets["test"]

    if len(traindata) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(traindata)), 3):
            logger.info(f"Sample {index} of the training set: {traindata[index]}.")

    logger.info(f"  Num examples in train = {len(traindata)}")
    logger.info(f"  Num examples in validation = {len(valdata)}")
    logger.info(f"  Num examples in test = {len(testdata)}")
    
    y_train = np.array([i["labels"] for i in traindata])
    print("0's and 1's in train:", len([i for i in y_train if i==0]), len([i for i in y_train if i==1]))

    if args.scale_loss:
        class_weights = class_weight.compute_class_weight('balanced', classes=[0,1], y=y_train)
    else:
        class_weights = None
    print("class weights:",class_weights)

    train_dataloader = DataLoader(dataset=traindata, 
                                  shuffle=args.shuffle, 
                                  batch_size=batch_size,
                                  #num_workers=0,
                                  worker_init_fn=seed_worker,
                                  generator=g) 
    

    val_dataloader = DataLoader(dataset=valdata, 
                                shuffle=args.shuffle, 
                                batch_size=batch_size,
                                #num_workers=0,
                                worker_init_fn=seed_worker,
                                generator=g)
    
    test_dataloader = DataLoader(dataset=testdata, 
                                shuffle=args.shuffle, 
                                batch_size=batch_size,
                                #num_workers=0,
                                worker_init_fn=seed_worker,
                                generator=g)
    
    print("Data prepared. Starting training...")
    
    trainer = seq_classification_trainer(accelerator=accelerator, 
                                         trainingargs=args, 
                                         model=model,
                                         tokenizer=tokenizer, 
                                         train_dataloader=train_dataloader, 
                                         val_dataloader=val_dataloader,
                                         test_dataloader=test_dataloader,
                                         class_weights=class_weights,
                                         logger=logger)


    trainer.train()


if __name__ == "__main__":
    main()