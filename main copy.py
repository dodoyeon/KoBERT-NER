import argparse
import os
from copy import deepcopy
import torch
from torch.optim import Adam

from chatgpt_ppo.ppo_trainer import PPOTrainer
from chatgpt_ppo.naive_strategy import NaiveStrategy

from utils_main import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP

from data_loader import load_and_cache_examples


# PPO
def main(args):
    init_logger()
    set_seed(args)

    config_class, model_class, token_class = MODEL_CLASSES[args.model_type]

    tokenizer = token_class.from_pretrained(args.model_name)
    

    # train_dataset = None
    # test_dataset = None

    model_config = config_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_config))
    actor = model_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_actor), config=model_config)
    critic = model_class.from_pretrained(args.model_name, config=model_config)
    tokenizer = token_class.from_pretrained(args.model_name)
    initial_model = deepcopy(actor)
    # reward = # human (me!)

    actor_optim = Adam(actor.parameters(), lr=5e-6)
    critic_optim = Adam(critic.parameters(), lr=5e-6)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 추가
    
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    strategy = NaiveStrategy()
    (actor, actor_optim), (critic, critic_optim), initial_model = strategy.prepare( # reward_model, 
    (actor, actor_optim), (critic, critic_optim), initial_model) # reward_model, 

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
        return {k: v.cuda() for k, v in batch.items()}

    trainer = PPOTrainer(strategy,
                     actor,
                     critic,
                    #  reward_model,
                     initial_model,
                     actor_optim,
                     critic_optim,
                     max_epochs=args.max_epochs,
                     train_batch_size=args.train_batch_size,
                     tokenizer=tokenize_fn,
                     max_length=128, # 이건 뭐고
                     do_sample=True,
                     temperature=1.0, # 이건 뭐야
                     top_k=50, # 얜 뭐임?
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id)

    ## train!
    trainer.fit(list_prompt,  # 입력 prompt
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
 
    # if args.do_eval:
    #     test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    # if args.do_train:
    #     train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    # trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    # if args.do_train:
    #     trainer.train()

    # if args.do_eval:
    #     trainer.load_model()
    #     trainer.evaluate("test", "eval")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_3_PPO', type=str, default='./data_kochatgpt/kochatgpt_3_PPO.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output_3_PPO')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--max_timesteps', type=int, default=3)
    parser.add_argument('--update_timesteps', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=250)

    parser.add_argument('--model_type', type=str, default='koelectra-base')
    parser.add_argument('--finetune_dir', type=str, default='./model_0418/')
    parser.add_argument('--finetune_actor', type=str, default='pytorch_model.bin')
    parser.add_argument('--finetune_config', type=str, default='config.json')


    args = parser.parse_args(args=[])
    # args = parser.parse_args()

    args.model_name = MODEL_PATH_MAP[args.model_type]
    main(args)
