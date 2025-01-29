def add_default_args(parser):

    parser.add_argument('-e', '--episode-number',
                        default=10000,
                        type=int,
                        help='Number of episodes')
    parser.add_argument('-l', '--learning-rate',
                        default=0.0001,
                        type=float,
                        help='Learning rate')
    parser.add_argument('-op', '--optimizer',
                        choices=['Adam', 'RMSProp'],
                        default='RMSProp',
                        help='Neural Network Optimization method')
    parser.add_argument('-m', '--memory-capacity',
                        default=100000,
                        type=int,
                        help='Memory capacity')
    parser.add_argument('-b', '--batch-size',
                        default=32,
                        type=int,
                        help='Batch size')
    parser.add_argument('-t', '--target-frequency',
                        default=10000,
                        type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration',
                        default=1000000,
                        type=int,
                        help='Maximum exploration step')
    parser.add_argument('-fsm', '--first-step-memory',
                        default=10000,
                        type=float,
                        help='Number of initial steps for just filling the memory')
    parser.add_argument('-rs', '--replay-steps',
                        default=4,
                        type=float,
                        help='Steps between updating the network')
    parser.add_argument('-nn', '--number-nodes',
                        default=32,
                        type=int,
                        help='Number of nodes in each layer of NN')
    parser.add_argument('-tt', '--target-type',
                        choices=['DQN', 'DDQN'],
                        default='DQN')
    parser.add_argument('-mt', '--memory',
                        choices=['UER', 'PER', 'SER'],
                        default='UER')
    parser.add_argument('-pl', '--prioritization-scale',
                        default=0.5,
                        type=float,
                        help='Scale for prioritization')
    parser.add_argument('-du', '--dueling',
                        action='store_true',
                        help='Enable Dueling architecture if "store_false"')
    parser.add_argument('-test', '--test',
                        action='store_true',
                        help='Enable the test phase if "store_false"')

    # Game Parameters
    parser.add_argument('-k', '--agents-number',
                        default=3,
                        type=int,
                        help='The number of agents')
    parser.add_argument('-v', '--visual-radius',
                        default=7,
                        type=int,
                        help='The visual radius of each agent')
    parser.add_argument('-ds', '--discount-state',
                        action='store_true',
                        help='Turn on discount mode if "store_false"')
    parser.add_argument('-d', '--discount-amount',
                        default=0.2,
                        type=int,
                        help='Amount of apple discounted from the stock at each step')
    parser.add_argument('-th', '--threshold',
                        default=1,
                        type=int,
                        help='threshold of apples in the stock allowing the agent to deposit one apple')
    parser.add_argument('-ms', '--max-stock',
                        default=20,
                        type=int,
                        help='maximum number of apples that can be present in the agents stock')
    parser.add_argument('-ts', '--max-timestep',
                        default=1000,
                        type=int,
                        help='Maximum number of steps per episode')

    parser.add_argument('-rm', '--max-random-moves',
                        default=0,
                        type=int,
                        help='Maximum number of random initial moves for agents')

    # Visualization Parameters
    parser.add_argument('-gf', '--generate-figures',
                        action='store_true',
                        help='if "store_false" enable to generate figures'
                             'from collected data"')
    parser.add_argument('-r', '--render',
                        action='store_true',
                        help='Turn on visualization if "store_false"')
    parser.add_argument('-re', '--recorder',
                        action='store_true',
                        help='Store the visualization as a movie if "store_false"')
