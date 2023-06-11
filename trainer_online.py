import os
import tqdm

from torch.utils.data import DataLoader
import torch

from utils_main import get_labels, custom_loss

class Trainer():
    def __init__(self, dataset, actor, actor_optimizer, initial_model, scheduler):
        self.train_data = dataset
        self.label_lst = get_labels(args)
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.model = actor
        self.initial_model = initial_model
        self.optimizer = actor_optimizer

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_step(self, input_ids):
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
        inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[3]}
        if self.args.model_type != 'distilkobert':
            inputs['token_type_ids'] = batch[2]
        outputs = self.model(**inputs)
        loss = outputs[0] # ??
        action_log_probs = self.model(input_ids, num_actions, attention_mask=attention_mask)
        base_action_log_probs = self.initial_model(input_ids)

        loss = custom_loss(action_log_probs,
                            base_action_log_probs,
                            action_mask=action_mask)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss 


    def train(self, args):
        self.model.train()
        self.model.to(self.device)
        
        if not args.online:
            tr_loss = 0

            dataloader = DataLoader(self.train_dataset, shuffle = True, batch_size = args.batch_size)
            for epoch in tqdm(range(args.epochs)):
                for step, batch in enumerate(dataloader):
                    loss = train_step(batch)
                    if step % args.logging_steps == 0:
                        if loss < tr_loss:
                            torch.save(actor.state_dict(),  os.path.join(args.output_dir, 'actor.pt'))
                            torch.save(actor_optimizer.state_dict(), os.path.join(args.output_dir, 'actor_optim_checkpoint_%d.pt' % (torch.cuda.current_device())))
        
        else:
            pass

    
    def eval(self, test_dataset):
        self.model.eval()
        dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.batch_size)
        eval_loss = 0
        nb_eval_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2] # ? 모델안에서 loss 나오나

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                # Slot prediction
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps # 원래 step 수로 나누나??
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
            if not os.path.exists(self.args.pred_dir):
                os.mkdir(self.args.pred_dir)

            with open(os.path.join(self.args.pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        # logger.info("***** Eval results *****")
        # for key in sorted(results.keys()):
        #     logger.info("  %s = %s", key, str(results[key]))
        # logger.info("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results