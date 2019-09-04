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
    sub = parser.add_subparsers()

    mode_train_cataloger = sub.add_parser('train:cataloger')
    mode_train_cataloger.add_argument('--input', type=Path, required=True)
    mode_train_cataloger.add_argument('--model_intent', type=Path,
                                      default="distributed-representation-superimposer/model/model_intent.pth")
    mode_train_cataloger.add_argument('--model_place', type=Path,
                                      default="distributed-representation-superimposer/model/model_place.pth")
    mode_train_cataloger.add_argument('--model_datetime', type=Path,
                                      default="distributed-representation-superimposer/model/model_datetime.pth")
    mode_train_cataloger.add_argument('--validation-only', action='store_true')
    mode_train_cataloger.set_defaults(handler=cmd_train_cataloger)

    mode_eval_cataloger = sub.add_parser('eval:cataloger')
    mode_eval_cataloger.add_argument('--text', type=str, required=True)
    mode_eval_cataloger.add_argument('--model_intent', type=Path,
                                     default="distributed-representation-superimposer/model/model_intent.pth")
    mode_eval_cataloger.add_argument('--model_place', type=Path,
                                     default="distributed-representation-superimposer/model/model_place.pth")
    mode_eval_cataloger.add_argument('--model_datetime', type=Path,
                                     default="distributed-representation-superimposer/model/model_datetime.pth")
    mode_eval_cataloger.set_defaults(handler=cmd_eval_cataloger)

    mode_train_superimposer = sub.add_parser('train:superimposer')
    mode_train_superimposer.add_argument('--input', type=Path, required=True)
    mode_train_superimposer.add_argument('--model', type=Path,
                                         default="distributed-representation-superimposer/model/model_superimposer.pth")
    mode_train_superimposer.add_argument('--model_intent', type=Path,
                                         default="distributed-representation-superimposer/model/model_intent.pth")
    mode_train_superimposer.add_argument('--model_place', type=Path,
                                         default="distributed-representation-superimposer/model/model_place.pth")
    mode_train_superimposer.add_argument('--model_datetime', type=Path,
                                         default="distributed-representation-superimposer/model/model_datetime.pth")
    mode_train_superimposer.add_argument('--validation-only', action='store_true')
    mode_train_superimposer.set_defaults(handler=cmd_train_superimposer)

    mode_eval_superimposer = sub.add_parser('eval:superimposer')
    mode_eval_superimposer.add_argument('--text1', type=str, required=True)
    mode_eval_superimposer.add_argument('--text2', type=str, required=True)
    mode_eval_superimposer.add_argument('--model', type=Path,
                                        default="distributed-representation-superimposer/model/model_superimposer.pth")
    mode_eval_superimposer.add_argument('--model_intent', type=Path,
                                        default="distributed-representation-superimposer/model/model_intent.pth")
    mode_eval_superimposer.add_argument('--model_place', type=Path,
                                        default="distributed-representation-superimposer/model/model_place.pth")
    mode_eval_superimposer.add_argument('--model_datetime', type=Path,
                                        default="distributed-representation-superimposer/model/model_datetime.pth")
    mode_eval_superimposer.set_defaults(handler=cmd_eval_superimposer)

    mode_create_all = sub.add_parser('create:all')
    mode_create_all.set_defaults(handler=cmd_create_all)

    mode_create_superimposer = sub.add_parser('create:superimposer')
    mode_create_superimposer.add_argument('--input', type=Path,
                                          default="distributed-representation-superimposer/data/data_cataloger.tsv")
    mode_create_superimposer.set_defaults(handler=cmd_create_superimposer)

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
