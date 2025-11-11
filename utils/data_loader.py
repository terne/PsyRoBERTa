import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import pandas as pd
from tqdm import tqdm

class ClinicalNotesDataset(Dataset):
    def __init__(self, X, y, ids, encounters, tokenizer, max_seq_len: int, docstride: int, uncased: bool, max_seq_splits: int):
        """
        X = list or array of strings (notes)
        y = relevant labels as integers 
        
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.docstride = docstride
        self.uncased = uncased
        self.max_seq_splits = max_seq_splits
        self.X = X
        self.y = y
        self.ids = ids
        self.encounters = encounters
        self.samples = self.make_dataset()
        

    def __len__(self):
        return len(self.samples)
    
    
    def make_dataset(self):
        """ 
        Function to tokenize the notes and return tensors.

        """
        
        input_ids = []
        attention_masks = []
        labs = []
        patientids = []
        encounterids = []
        
        for x,y, patient_id, encounter_id in tqdm(list(zip(self.X, self.y, self.ids, self.encounters))):
            if self.uncased:
                x=x.lower()
            encoded_dict = self.tokenizer.encode_plus(
                x,  # Sentence to encode
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_seq_len,  # Pad & truncate all sentences.
                padding="max_length", #(needing to specify truncation=True depends on version)
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                stride=self.docstride, # The stride used when the context is too large and is split across several features.
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt'  # Return pytorch tensors.
            )
            
            for inputs, attentions in list(zip(encoded_dict['input_ids'],encoded_dict['attention_mask']))[:self.max_seq_splits]:
                #print(i.shape)
                # Add the encoded sentence to the list.
                input_ids.append(inputs)
                #And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(attentions)
                labs.append(y)
                patientids.append(patient_id)
                encounterids.append(encounter_id)

        # Convert the lists into tensors.
        #input_ids_ = torch.cat(input_ids, dim=0).type(torch.LongTensor)
        input_ids_ = torch.stack((input_ids), dim=0).type(torch.LongTensor)
        #attention_masks = torch.cat(attention_masks, dim=0).type(torch.LongTensor)
        attention_masks = torch.stack((attention_masks),dim=0).type(torch.LongTensor)
        labels = torch.tensor(labs).type(torch.LongTensor)
        patientids = torch.tensor(patientids)
        enounterids = torch.tensor(encounterids)


        #print(len(input_ids_), len(attention_masks), len(labels))
        assert len(input_ids_) == len(attention_masks) == len(labels) == len(patientids) == len(encounterids)
        #samples = tuple(zip(input_ids, attention_masks, labels, sentence_id))
        samples = tuple(zip(input_ids_, attention_masks, labels, patientids, encounterids))

        return samples


    def __getitem__(self, idx) -> Dict[str, Any]:
            """
            Args:
                idx (int): Index

            Returns:
                Dictionary with keys: (inputs, attn_masks, labels)
                where the input_ids is the tokenized sentence,
                attention_masks the tokenizer att mask,
                labels are the target
            """
            #input_ids, attention_masks, labels, note_ids = self.samples[idx]
            input_ids, attention_masks, labels, patientids, encounterids = self.samples[idx]
            return {"inputs": input_ids,
                    "attn_masks": attention_masks,
                    "labels": labels,
                    "patient_id": patientids,
                    "encounter_id": encounterids
                    }