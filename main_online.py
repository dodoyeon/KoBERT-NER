import os
import argparse
from copy import deepcopy
from datetime import datetime

from trainer_online import Trainer_online

from utils_main import init_logger, set_seed, MODEL_CLASSES, MODEL_PATH_MAP

from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)

    config_class, model_class, token_class = MODEL_CLASSES[args.model_type]
    tokenizer = token_class.from_pretrained(args.model_name_or_path)

    model_config = config_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_config))
    actor = model_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_actor), config=model_config)
    initial_model = deepcopy(actor)

    
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        args.output_dir += datetime.today().strftime('_%Y%m%d_%H%M%S')
        os.makedirs(args.output_dir)
        print('Output directory is CHANGED to avoid overlapping.')

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    else:
        test_dataset = None
    
    ## train!
    trainer = Trainer_online(args, train_dataset, test_dataset, actor, initial_model)
    trainer.train()

    ## eval
    # if args.do_eval:
    #     trainer.eval(args, test_dataset)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_dir', default='/model_chpt', help='save model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--pretrain', type=str, default=False)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=20, type=float, help="Total number of training epochs to perform.") # 20
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.") # 1e-8
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--write_pred", default=True, action="store_true", help="Write prediction during evaluation")

    parser.add_argument('--model_type', type=str, default='koelectra-base')
    parser.add_argument('--finetune_dir', type=str, default='./model_0418/')
    parser.add_argument('--finetune_actor', type=str, default='pytorch_model.bin')
    parser.add_argument('--finetune_config', type=str, default='config.json')
    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    
    parser.add_argument("--do_eval", default = True, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")

    parser.add_argument("--loss", default='online', choices=['base', 'online'], help="Loss selection")
    parser.add_argument("--online", default=False, help ='whether online learning or not')
    parser.add_argument("--beta", default=0.03, help ='kl loss hyper parameter') # 0.03

    args = parser.parse_args(args=[])
    # args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)