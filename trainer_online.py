import os
import numpy as np
from tqdm import tqdm, trange
import logging

from torch.optim import Adam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt

from utils_main import compute_metrics, show_report, get_test_texts, custom_loss

logger = logging.getLogger(__name__)

class Trainer_online():
    def __init__(self, args, train_dataset, test_dataset, actor, initial_model, label_lst):
        self.args = args
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.label_lst = label_lst
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

        self.model = actor
        self.initial_model = initial_model

        self.dataloader = DataLoader(self.train_data, shuffle = True, batch_size = args.batch_size)

        if self.args.max_steps > 0:
            t_total = args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(self.dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.dataloader) // args.gradient_accumulation_steps * args.epochs


        self.optimizer = Adam(actor.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # self.optimizer = AdamW(actor.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.initial_model.to(self.device)

        self.test_texts = None
        if args.write_pred:
            self.test_texts = get_test_texts(args)

        self.lrs = []

    def train_step(self, batch, step):
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
        inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[3]}
        if self.args.model_type != 'distilkobert':
            inputs['token_type_ids'] = batch[2]

        outputs = self.model(**inputs)
        
        if self.args.loss == 'base':
            loss = outputs[0]

        elif self.args.loss == 'online':
            base_outputs = self.initial_model(**inputs)
            
            loss = custom_loss(self.args.beta,
                                outputs,
                                base_outputs[1],
                                inputs['attention_mask'])
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
            self.optimizer.zero_grad()
        return loss 


    def train(self):
        self.model.train()
        self.model.to(self.device)
        
        if not self.args.online:
            tr_loss = 0
            global_step = 0

            for epoch in trange(self.args.epochs, desc='Epoch'):
                for step, batch in enumerate(self.dataloader):
                    loss = self.train_step(batch, step)
                    tr_loss += loss.item()
                    global_step += 1
                    
                    # if self.args.do_eval and global_step % self.args.logging_steps == 0:
                    #     assert self.test_data != None, "Test Data Not Found"
                self.eval(global_step)

                    # if global_step % self.args.save_steps == 0:
            self.save_model()
                        # torch.save(self.model.state_dict(),  os.path.join(self.args.output_dir, 'actor.pt'))
                        # torch.save(self.optimizer.state_dict(), os.path.join(self.args.output_dir, 'actor_optim_checkpoint_%d.pt' % (torch.cuda.current_device())))

            # plt.plot(self.lrs) # 왠지 모를 에러 때문에 포기 (그냥 list 로 봣다)
            # plt.xlabel('epochs')
            # plt.ylabel('learning rate')
            # plt.title('Learning rate scheduling')
            # plt.show()
        
        else:
            pass

        return tr_loss / global_step, global_step

    
    def eval(self, step):
        dataloader = DataLoader(self.test_data, shuffle = False, batch_size = self.args.batch_size)
        eval_loss = 0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None


        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                # Slot prediction
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps 
            results = {
                "loss": eval_loss
            }

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.args.write_pred:

            with open(os.path.join(self.args.output_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results
    
    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.output_dir)

        # Save training arguments and label list together with trained model
        param_dict = {'training_args': self.args,
                      'label_lst': self.label_lst}

        torch.save(param_dict, os.path.join(self.args.output_dir, 'training_params.bin'))

        # logger.info("Saving model checkpoint to %s", self.args.model_dir)