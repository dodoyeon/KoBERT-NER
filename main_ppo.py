import argparse
import os
import csv
from copy import deepcopy
import torch
from torch.optim import Adam

from chat_ppo.ppo_trainer import PPOTrainer
from chat_ppo.naive_strategy import NaiveStrategy
from chat_ppo.reward_mari import reward_algorithm

from utils_main import init_logger, load_tokenizer, get_labels, set_seed, MODEL_CLASSES, MODEL_PATH_MAP

from data_loader import load_and_cache_examples



# PPO
def main(args):
    init_logger()
    set_seed(args)

    config_class, model_class, token_class = MODEL_CLASSES[args.model_type]
    label_lst = get_labels(args)
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model_config = config_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_config))
    actor = model_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_actor), config=model_config)
    critic = reward_algorithm
    reward_model = reward_algorithm # human (me!)
    tokenizer = token_class.from_pretrained(args.model_name_or_path)
    initial_model = deepcopy(actor)
    

    actor_optim = Adam(actor.parameters(), lr=5e-5)
    # critic_optim = Adam(critic.parameters(), lr=5e-6)

    
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(args.data_path_3_PPO) as f:
    #     readdata = list(csv.reader(f, delimiter='\t'))
    #     list_data = [i[0] for i in readdata]
    #     list_label = [i[1] for i in readdata]

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    # def tokenize_fn(texts):
    #     batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
    #     return {k: v.cuda() for k, v in batch.items()}

    strategy = NaiveStrategy()
    (actor, actor_optim), initial_model = strategy.prepare( # (critic, critic_optim), reward_model, 
    (actor, actor_optim), initial_model) # (critic, critic_optim), reward_model, 


    trainer = PPOTrainer(strategy,
                     actor,
                     critic,
                     reward_model,
                     initial_model,
                     actor_optim,
                    #  critic_optim,
                     max_epochs=args.max_epochs,
                     train_batch_size=args.train_batch_size,
                    #  tokenizer=tokenizer,
                     max_length=128, # 이건 뭐고
                     do_sample=True,
                     temperature=1.0, # 이건 뭐야
                     top_k=50, # 얜 뭐임?
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id)

    ## train!
    trainer.fit(train_dataset, # 입력 prompt
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)

    ## save
    # save model checkpoint after fitting on only rank0
    strategy.save_model(actor, os.path.join(args.output_dir, 'actor.pt'), only_rank0=True)
    # save optimizer checkpoint on all ranks
    strategy.save_optimizer(actor_optim,
                            os.path.join(args.output_dir, 'actor_optim_checkpoint_%d.pt' % (torch.cuda.current_device())),
                            only_rank0=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument('--output_dir', type=str, default='./output_3_PPO')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--max_timesteps', type=int, default=3)
    parser.add_argument('--update_timesteps', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=250) # ?? max_seq_len 이랑 다른지 확인

    parser.add_argument('--model_type', type=str, default='koelectra-base')
    parser.add_argument('--finetune_dir', type=str, default='./model_0418/')
    parser.add_argument('--finetune_actor', type=str, default='pytorch_model.bin')
    parser.add_argument('--finetune_config', type=str, default='config.json')

    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    parser.add_argument("--do_train", default = True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default = False, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")

    args = parser.parse_args(args=[])
    # args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
