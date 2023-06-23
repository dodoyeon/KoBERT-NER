import os
import argparse
from copy import deepcopy
from datetime import datetime
from transformers import *
import torch
import logging

from trainer_online import Trainer_online

from utils_main import init_logger, set_seed, get_labels, MODEL_CLASSES, MODEL_PATH_MAP

from data_loader import load_and_cache_examples

def load_onlinemodel(args, config_class, model_class, label_lst, device):
    config = config_class.from_pretrained(args.model_name_or_path,
                                                num_labels=len(label_lst),
                                                finetuning_task=args.task,
                                                id2label={str(i): label for i, label in enumerate(label_lst)},
                                                label2id={label: i for i, label in enumerate(label_lst)})
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    chkpt = torch.load(os.path.join(args.finetune_dir, args.finetune_actor))
    # 이 부분 자동화로 바꿔야하고 initialization 도 더 효율적으로 생각해보기
    # chkpt['classifier.weight'] = torch.cat((chkpt['classifier.weight'], torch.randn(2, 768).to(device)))
    # chkpt['classifier.bias'] = torch.cat((chkpt['classifier.bias'], torch.randn(2).to(device)))
    chkpt['classifier.weight'] = torch.cat((chkpt['classifier.weight'], torch.zeros(2, 768).to(device)))
    chkpt['classifier.bias'] = torch.cat((chkpt['classifier.bias'], torch.zeros(2).to(device)))
    model.load_state_dict(chkpt) # strict = False
    return model


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        args.output_dir += datetime.today().strftime('_%Y%m%d_%H%M%S')
        os.makedirs(args.output_dir)
        print('Output directory is CHANGED to avoid overlapping.')
    
    init_logger(args)

    logger = logging.getLogger(__name__)

    logger.info(args)
    set_seed(args)

    config_class, model_class, token_class = MODEL_CLASSES[args.model_type]
    tokenizer = token_class.from_pretrained(args.model_name_or_path)
    label_lst = get_labels(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.aftertask:
        actor = load_onlinemodel(args, config_class, model_class, label_lst, device)
        initial_model = None
        
    else:
        model_config = config_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_config))
        actor = model_class.from_pretrained(os.path.join(args.finetune_dir, args.finetune_actor), config=model_config)
        initial_model = deepcopy(actor)
    
    print(args)
    
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    
    if args.loss == 'ewc':
        replay_dataset = load_and_cache_examples(args, tokenizer, mode="dev") # temporal
    else:
        replay_dataset = None

    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    else:
        test_dataset = None
    
    ## train!
    trainer = Trainer_online(args, train_dataset, replay_dataset, test_dataset, actor, initial_model, label_lst, device)
    trainer.train()

    ## eval
    # if args.do_eval:
        # trainer.eval(args, test_dataset)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument('--output_dir', type=str, default='./output')
    # parser.add_argument('--model_dir', default='/model_chkpt', help='save model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--pretrain', type=str, default=False)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.") # ADAM 5e-5 / 1e-3
    parser.add_argument("--epochs", default=50, type=float, help="Total number of training epochs to perform.") # 20
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
    parser.add_argument('--finetune_dir', type=str, default='./model/') # output_20230620_163122
    parser.add_argument('--finetune_actor', type=str, default='pytorch_model.bin')
    parser.add_argument('--finetune_config', type=str, default='config.json')
    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    
    parser.add_argument("--do_train", default = True, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval", default = True, action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--train_file", default="new_fine.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="te_original.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="tr_50.tsv", type=str, help="Test file")
    
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")

    parser.add_argument("--loss", default='ewc', choices=['base', 'kl', 'ewc'], help="Loss selection")
    parser.add_argument("--aftertask", default=True, help ='whether online learning starts or not')
    parser.add_argument("--beta", default=0.03, help ='kl loss hyper parameter') # 0.03, 0.05

    args = parser.parse_args(args=[])
    # args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)