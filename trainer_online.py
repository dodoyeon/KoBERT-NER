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
        action_log_probs = self.model(input_ids, num_actions, attention_mask=attention_mask)
        base_action_log_probs = self.initial_model(input_ids)

        loss = custom_loss(action_log_probs,
                            base_action_log_probs,
                            action_mask=action_mask)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    def train(self, args):
        self.model.train()
        self.model.to(self.device)
        
        if not args.online:
            tr_loss = 0

            dataloader = DataLoader(self.train_dataset, shuffle = True, batch_size = args.batch_size)
            for epoch in tqdm(range(args.epochs)):
                for step, batch in enumerate(dataloader):
                    train_step(batch)
                    if step % args.logging_steps == 0:
                        if loss < tr_loss:
                            torch.save(actor.state_dict(),  os.path.join(args.output_dir, 'actor.pt'))
                            torch.save(actor_optimizer.state_dict(), os.path.join(args.output_dir, 'actor_optim_checkpoint_%d.pt' % (torch.cuda.current_device())))
        
        else:
            pass

    
    def eval(self, test_dataset):
        self.model.eval()
        dataloader = DataLoader(test_dataset, shuffle = False, batch_size = args.batch_size)
        with torch.no_grad():
            for batch in dataloader:
                pass