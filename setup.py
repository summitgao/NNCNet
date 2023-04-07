import argparse


def option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-b',
                        '--batch-size',
                        default=16,
                        type=int,
                        metavar='N')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.0005,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--schedule',
                        default=[120, 160],
                        nargs='*',
                        type=int,
                        help='learning rate schedule (when to drop lr by 10x)')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    parser.add_argument('--dim',
                        default=128,
                        type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument(
        '-k',
        default=2048,
        type=int,
        help='queue size; number of negative keys (default: 2048)')
    parser.add_argument('-m',
                        default=0.9,
                        type=float,
                        help='momentum of updating key encoder (default: 0.9)')
    parser.add_argument('-t',
                        default=0.07,
                        type=float,
                        help='softmax temperature (default: 0.07)')

    parser.add_argument('--pretrained',
                        default='',
                        type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('-p',
                        '--print-freq',
                        default=100,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--cos',
                        action='store_true',
                        help='use cosine lr schedule')

    args = parser.parse_args()

    return args
