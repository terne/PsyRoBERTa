from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, get_scheduler
import torch
import torch.nn as nn
from torch.optim import AdamW # or use transformers version?
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
#from tqdm import tqdm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import numpy as np
import csv
import os
import math

class seq_classification_trainer:
    def __init__(self, accelerator, trainingargs, model, tokenizer, train_dataloader, val_dataloader, test_dataloader, class_weights, logger):

        self.trainingargs = trainingargs
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.accelerator = accelerator
        self.seed = trainingargs.random_seed
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights).cuda()
        else:
            self.class_weights = class_weights
        self.logger = logger

        
        self.num_train_epochs = self.trainingargs.num_epochs
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.trainingargs.gradient_accumulation_steps)
        num_training_steps = self.num_train_epochs * num_update_steps_per_epoch

        if self.trainingargs.warmup_steps:
            warmup_steps = self.trainingargs.warmup_steps
        else:
            warmup_steps = int(num_training_steps*self.trainingargs.warmup_proportion)

        self.optimizer = AdamW(self.model.parameters(), lr=self.trainingargs.lr) # default weight_decay=0.01 for all params. Could be improved.
        
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer=self.optimizer,
                            patience=5, # Note that LR will be stable, i.e. not reduce, when we only run 5 epochs. 
                            #patience=2, # possible alternative, however LR is already quite low.
                            factor=0.1,
                            verbose=True
                            )
        #self.lr_scheduler = get_linear_schedule_with_warmup(
        #            optimizer=self.optimizer,
        #            num_warmup_steps=warmup_steps,
        #            num_training_steps=self.num_training_steps
        #        )
        
        
        

    def train(self):

        model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.test_dataloader, self.lr_scheduler
        )

        # recalculate steps after .prepare since length of dataloader might have changed
        num_train_epochs = self.trainingargs.num_epochs
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.trainingargs.gradient_accumulation_steps)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        total_batch_size = self.trainingargs.batch_size * self.accelerator.num_processes * self.trainingargs.gradient_accumulation_steps
        
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num Epochs = {num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.trainingargs.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.trainingargs.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {num_training_steps}")

        
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        #if overrode_max_train_steps:
        #args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        #args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        checkpointing_steps = self.trainingargs.checkpointing_steps
        completed_steps = 0
        progress_bar = tqdm(range(num_training_steps), disable=not self.accelerator.is_main_process)


        # Register the LR scheduler
        self.accelerator.register_for_checkpointing(lr_scheduler)

        #completed_steps = 0
        starting_epoch = 0
        if self.trainingargs.resume_from_checkpoint:
            if self.trainingargs.resume_from_checkpoint is not None or self.trainingargs.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.trainingargs.resume_from_checkpoint}")
                self.accelerator.load_state(self.trainingargs.resume_from_checkpoint)
                path = os.path.basename(self.trainingargs.resume_from_checkpoint)
                training_difference = os.path.splitext(path)[0]
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1


        self.accelerator.print(f'TRAINING STEPS: {num_training_steps}')
        
        if self.trainingargs.with_tracking:
            experiment_config = vars(self.trainingargs)
            # TensorBoard cannot log Enums, need the raw value
            #experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("finetune", experiment_config)
        

        train_losses = []
        for epoch in range(starting_epoch, self.num_train_epochs):
            epoch_trainlosses = []
            
            results = {
                        "epoch": epoch,
                        "completed_steps": 0,

                        "train_predictions": [], 
                        "val_predictions": [],
                        "test_predictions": [],

                        "train_targets": [],
                        "val_targets": [],
                        "test_targets": [],

                        "train_probs": [],
                        "val_probs": [],
                        "test_probs": [],

                        "train_pids" : [],
                        "val_pids" : [],
                        "test_pids" : [],

                        "train_eids" : [],
                        "val_eids" : [],
                        "test_eids" : [],

                        "train_loss": 0.0,
                        "val_loss": 0.0,
                        "test_loss": 0.0,

                        "train_f1": 0.0,
                        "train_mcc": 0.0,

                        "val_f1": 0.0,
                        "val_mcc": 0.0,
                        "ppv": 0.0,

                        "tn": 0.0,
                        "fp": 0.0,
                        "fn": 0.0,
                        "tp": 0.0,
                }

            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model): 
                    inputs, attn, targets = batch["inputs"],batch["attn_masks"], batch["labels"]
                    #outputs = model(inputs)
                    outputs = model(inputs,attention_mask=attn,labels=targets)
                    #loss = outputs.loss
                    loss = self.criterion(outputs['logits'], targets)
                    total_loss += loss.detach().float()
                    self.accelerator.log({"training_loss": float(loss.data.detach().cpu())}, step=completed_steps)
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                

                probabilities = nn.functional.softmax(outputs.logits, dim=-1)
                probabilities, references = self.accelerator.gather_for_metrics((probabilities,batch["labels"]))
                pos_probs = probabilities[:,1:].flatten()
                results["train_probs"].extend(pos_probs.detach().cpu().tolist())
                predictions = np.argmax(probabilities.detach().cpu(), axis=1).flatten()
                
                results["train_predictions"].extend(predictions.tolist())
                results["train_targets"].extend(references.detach().cpu().tolist())
                pid, eid = self.accelerator.gather_for_metrics((batch["patient_id"], batch["encounter_id"]))

                results["train_pids"].extend(pid.detach().cpu().tolist())
                results["train_eids"].extend(eid.detach().cpu().tolist())

                epoch_trainlosses.append(float(loss.data.detach().cpu().numpy()))

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if self.trainingargs.checkpoint_dir is not None:
                            output_dir = os.path.join(self.trainingargs.checkpoint_dir, output_dir)
                        self.accelerator.save_state(output_dir)
                
            mean_loss = np.mean(epoch_trainlosses)
            train_losses.append(mean_loss)
            results["train_loss"] = mean_loss
            #results["total_train_loss_mean"] = total_loss.item() / len(train_dataloader)
            trainf1 = f1_score(results["train_targets"],results["train_predictions"])
            trainmcc = matthews_corrcoef(results["train_targets"],results["train_predictions"])
            results["train_f1"] = trainf1
            results["train_mcc"] = trainmcc

            
            
            # compute performance on validation set
            model.eval()
            epoch_vallosses = []
            for step, batch in enumerate(val_dataloader):
                inputs, attn, targets = batch["inputs"],batch["attn_masks"], batch["labels"]
                #pid, eid = batch["patient_id"], batch["encounter_id"]
                with torch.no_grad():
                    outputs = model(inputs, attention_mask=attn, labels=targets)
                
                val_loss = self.criterion(outputs['logits'], targets)#outputs.loss

                probabilities = nn.functional.softmax(outputs.logits, dim=-1)
                probabilities, references = self.accelerator.gather_for_metrics((probabilities,batch["labels"]))
                
                if step==0:
                    print(probabilities.detach().cpu()[:10])
                predictions = np.argmax(probabilities.detach().cpu(), axis=1).flatten()
                pos_probs = probabilities[:,1:].flatten()
                results["val_probs"].extend(pos_probs.detach().cpu().tolist())
                
                results["val_predictions"].extend(predictions.tolist())
                results["val_targets"].extend(references.detach().cpu().tolist())

                pid, eid = self.accelerator.gather_for_metrics((batch["patient_id"], batch["encounter_id"]))

                results["val_pids"].extend(pid.detach().cpu().tolist())
                results["val_eids"].extend(eid.detach().cpu().tolist())

                epoch_vallosses.append(float(val_loss.data.detach().cpu().numpy()))
            
            
            mean_valloss = np.mean(epoch_vallosses)
            lr_scheduler.step(mean_valloss)
            results["val_loss"] = mean_valloss
            

            f1 = f1_score(results["val_targets"],results["val_predictions"])
            mcc = matthews_corrcoef(results["val_targets"],results["val_predictions"])
            results["val_f1"] = f1
            results["val_mcc"] = mcc
            tn, fp, fn, tp = confusion_matrix(results["val_targets"],results["val_predictions"]).ravel()
            results["tn"] = int(tn)
            results["fp"] = int(fp)
            results["fn"] = int(fn)
            results["tp"] = int(tp)
            results["ppv"] = float(tp/(tp+fp))
            results["completed_steps"] = completed_steps

            epoch_testlosses = []
            for step, batch in enumerate(test_dataloader):
                inputs, attn, targets = batch["inputs"],batch["attn_masks"], batch["labels"]
                #pid, eid = batch["patient_id"], batch["encounter_id"]
                with torch.no_grad():
                    outputs = model(inputs, attention_mask=attn, labels=targets)
                
                test_loss = self.criterion(outputs['logits'], targets)#outputs.loss

                probabilities = nn.functional.softmax(outputs.logits, dim=-1)
                probabilities, references = self.accelerator.gather_for_metrics((probabilities,batch["labels"]))
                
                #if step==0:
                #    print(probabilities.detach().cpu()[:10])
                predictions = np.argmax(probabilities.detach().cpu(), axis=1).flatten()
                pos_probs = probabilities[:,1:].flatten()
                results["test_probs"].extend(pos_probs.detach().cpu().tolist())
                
                results["test_predictions"].extend(predictions.tolist())
                results["test_targets"].extend(references.detach().cpu().tolist())

                pid, eid = self.accelerator.gather_for_metrics((batch["patient_id"], batch["encounter_id"]))

                results["test_pids"].extend(pid.detach().cpu().tolist())
                results["test_eids"].extend(eid.detach().cpu().tolist())

                epoch_testlosses.append(float(test_loss.data.detach().cpu().numpy()))
            
            mean_testloss = np.mean(epoch_testlosses)
            results["test_loss"] = mean_testloss

            exclude_in_mlflow = {"val_pids", "test_pids", "val_eids","test_eids","train_pids", "train_eids", "val_targets", "test_targets", "val_predictions", "test_predictions", "val_probs", "test_probs", "train_targets", "train_predictions", "train_probs"}
            if self.trainingargs.with_tracking:
                self.accelerator.log(
                    {x: results[x] for x in results if x not in exclude_in_mlflow},
                    step=completed_steps,
                )
            
            self.accelerator.print(f"epoch {epoch}: F1={f1}, MCC={mcc}")
            
            output_dir = f"epoch_{epoch}"
            output_dir = os.path.join(self.trainingargs.checkpoint_dir, output_dir)

            self.accelerator.save_state(output_dir)
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                with open(self.trainingargs.checkpoint_dir+'/results.csv','a') as f:
                    writer=csv.writer(f)
                    if epoch==starting_epoch:
                        writer.writerow(list(results.keys())) 
                    writer.writerow(list(results.values()))
                
        
        self.save_model(model)
        self.plot_loss(train_losses)
        if self.trainingargs.with_tracking:
            self.accelerator.end_training()


    
    def plot_loss(self, losses: List[float]):
        fig, ax = plt.subplots()
        sns.scatterplot(x=range(len(losses)), y=losses, ax=ax)
        sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
        ax.set_xticks(np.arange(self.num_train_epochs))
        ax.set_xticklabels([str(int(i)) for i in np.arange(1,self.num_train_epochs+1)])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.savefig(self.trainingargs.checkpoint_dir+"/loss_curve.png") # add more info


    
    def save_model(self, model):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            self.trainingargs.checkpoint_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.trainingargs.checkpoint_dir)
        
        