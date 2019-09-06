from argparse import ArgumentParser
from pathlib import Path
from cmd_train_cataloger import cmd_train_cataloger
from cmd_eval_cataloger import cmd_eval_cataloger
from cmd_train_superimposer import cmd_train_superimposer
from cmd_eval_superimposer import cmd_eval_superimposer
from cmd_create_data import cmd_create_all
from cmd_create_data import cmd_create_superimposer


def cmd_help(args):
    print(parser.parse_args([args.command, '--help']))


def main():
    parser = ArgumentParser()
    subparser = parser.add_subparsers()

    def mode_train_cataloger():
        sub = subparser.add_parser('train:cataloger')
        sub.add_argument('--input', type=Path, required=True)
        sub.add_argument('--model_intent', type=Path,
                         default="dr-superimposer/model/model_intent.pth")
        sub.add_argument('--model_place', type=Path,
                         default="dr-superimposer/model/model_place.pth")
        sub.add_argument('--model_datetime', type=Path,
                         default="dr-superimposer/model/model_datetime.pth")
        sub.add_argument('--validation-only', action='store_true')
        sub.set_defaults(handler=cmd_train_cataloger)
    mode_train_cataloger()

    def mode_eval_cataloger():
        sub = subparser.add_parser('eval:cataloger')
        sub.add_argument('--text', type=str, required=True)
        sub.add_argument('--model_intent', type=Path,
                         default="dr-superimposer/model/model_intent.pth")
        sub.add_argument('--model_place', type=Path,
                         default="dr-superimposer/model/model_place.pth")
        sub.add_argument('--model_datetime', type=Path,
                         default="dr-superimposer/model/model_datetime.pth")
        sub.set_defaults(handler=cmd_eval_cataloger)
    mode_eval_cataloger()

    def mode_train_superimposer():
        sub = subparser.add_parser('train:superimposer')
        sub.add_argument('--input', type=Path, required=True)
        sub.add_argument('--model', type=Path,
                         default="dr-superimposer/model/model_superimposer.pth")
        sub.add_argument('--model_intent', type=Path,
                         default="dr-superimposer/model/model_intent.pth")
        sub.add_argument('--model_place', type=Path,
                         default="dr-superimposer/model/model_place.pth")
        sub.add_argument('--model_datetime', type=Path,
                         default="dr-superimposer/model/model_datetime.pth")
        sub.add_argument('--validation-only', action='store_true')
        sub.set_defaults(handler=cmd_train_superimposer)
    mode_train_superimposer()

    def mode_eval_superimposer():
        sub = subparser.add_parser('eval:superimposer')
        sub.add_argument('--text1', type=str, required=True)
        sub.add_argument('--text2', type=str, required=True)
        sub.add_argument('--model', type=Path,
                         default="dr-superimposer/model/model_superimposer.pth")
        sub.add_argument('--model_intent', type=Path,
                         default="dr-superimposer/model/model_intent.pth")
        sub.add_argument('--model_place', type=Path,
                         default="dr-superimposer/model/model_place.pth")
        sub.add_argument('--model_datetime', type=Path,
                         default="dr-superimposer/model/model_datetime.pth")
        sub.set_defaults(handler=cmd_eval_superimposer)
    mode_eval_superimposer()

    def mode_create_all():
        sub = subparser.add_parser('create:all')
        sub.set_defaults(handler=cmd_create_all)
    mode_create_all()

    def mode_create_superimposer():
        sub = subparser.add_parser('create:superimposer')
        sub.add_argument('--input', type=Path,
                         default="dr-superimposer/data/data_cataloger.tsv")
        sub.set_defaults(handler=cmd_create_superimposer)
    mode_create_superimposer()

    def mode_help():
        sub = subparser.add_parser('help')
        sub.add_argument('cmd')
        sub.set_defaults(handler=cmd_help)
    mode_help()

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
