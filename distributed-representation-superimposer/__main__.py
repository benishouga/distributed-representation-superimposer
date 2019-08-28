from argparse import ArgumentParser
from pathlib import Path
from cmd_train import cmd_train
from cmd_eval import cmd_eval


def cmd_help(args):
    print(parser.parse_args([args.command, '--help']))


def main():
    parser = ArgumentParser()
    sub = parser.add_subparsers()

    mode_train = sub.add_parser('train')
    mode_train.add_argument('--input', type=Path, required=True)
    mode_train.add_argument('--model_intent', type=Path)
    mode_train.add_argument('--model_place', type=Path)
    mode_train.add_argument('--model_datetime', type=Path)
    mode_train.set_defaults(handler=cmd_train)

    mode_eval = sub.add_parser('eval')
    mode_eval.add_argument('--text', type=str, required=True)
    mode_eval.add_argument('--model_intent', type=Path, required=True)
    mode_eval.add_argument('--model_place', type=Path, required=True)
    mode_eval.add_argument('--model_datetime', type=Path, required=True)
    mode_eval.set_defaults(handler=cmd_eval)

    mode_help = sub.add_parser('help')
    mode_help.add_argument('cmd')
    mode_help.set_defaults(handler=cmd_help)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
