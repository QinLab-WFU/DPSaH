import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50_frozen_normalize", help="see _network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=120, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=112, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="adam", help="adam/rmsprop/adamw/sgd")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=4e-4, help="weight decay")

    # will be changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=24, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=16, help="length of hashing binary")

    # special settings

    ### Schedular
    parser.add_argument("--gamma", default=0.3, type=float, help="Learning rate reduction after tau epochs.")
    parser.add_argument("--tau", default=[80], nargs="+", type=int, help="Step size before reducing learning rate.")

    ### MarginLoss
    parser.add_argument("--margin", default=0.2, type=float, help="Margin for Triplet Loss")

    parser.add_argument(
        "--lr-beta", default=0.0005, type=float, help="Learning Rate for class margin parameters in MarginLoss"
    )
    parser.add_argument(
        "--wd-beta", default=0.0, type=float, help="Weight decay for class margin parameters in MarginLoss"
    )

    parser.add_argument("--beta", default=1.2, type=float, help="Initial Class Margin Parameter in Margin Loss")
    parser.add_argument("--nu", default=0, type=float, help="Regularisation value on betas in Margin Loss.")
    parser.add_argument("--beta_constant", action="store_true")

    ##### Evaluation Settings
    parser.add_argument("--k_vals", nargs="+", default=[1, 2, 4, 8], type=int, help="Recall @ Values.")

    ### Policy
    parser.add_argument(
        "--train_val_split",
        default=0.9,
        type=float,
        help="Percentage of training data that is retained for training. The remainder is used to set up the validation set.",
    )

    parser.add_argument(
        "--n_bins",
        default=30,
        type=int,
        help="Resolution of the support grid, i.e. number of bins between the support interval --bin_limit, K in paper.",
    )

    parser.add_argument(
        "--bin_limit",
        nargs="+",
        default=[0.1, 1.4],
        type=float,
        help="Limit of the sampling distribution support. Excluding small values removes negatives that might be too hard.",
    )

    parser.add_argument("--lr-policy", default=0.01, type=float, help="Learning rate of sampling policy.")
    parser.add_argument("--wd-policy", default=0.0, type=float, help="Weight decay of sampling policy.")

    parser.add_argument(
        "--policy_update_freq",
        default=30,
        type=int,
        help="Number of iteration (M in paper) to update the network before computing the validation reward metrics and updating the policy",
    )

    parser.add_argument(
        "--policy_action_space",
        nargs="+",
        default=[0.8, 1, 1.25],
        type=float,
        help="[alpha,1,beta] - values to update the sampling distribution bins by. Updates are done multiplicatively.",
    )

    parser.add_argument(
        "--policy_size",
        nargs="+",
        default=[128, 128],
        type=int,
        help="Size of the utilized policy. Values in the list denote number of neurons, length the number of layers.",
    )

    ###
    parser.add_argument("--policy_run_avgs_no_avg", action="store_true")
    parser.add_argument(
        "--policy_run_avgs",
        nargs="+",
        default=[2, 8, 16, 32],
        type=int,
        help="Running averages of state metrics defined in --policy_state_metrics to be included in the policy input state.",
    )

    parser.add_argument(
        "--policy_state_metrics",
        nargs="+",
        default=["recall", "nmi", "dists"],
        type=str,
        help="Metrics to include into the policy input state. Available options: recall, nmi & dists (intra- and interclass distances).",
    )

    ###
    parser.add_argument(
        "--policy_reward_metrics",
        nargs="+",
        default=["recall", "nmi"],
        type=str,
        help="Target reward metrics to be optimized on the validation set.",
    )

    parser.add_argument(
        "--state_history_m",
        default=20,
        type=int,
        help="Number of validation metrics steps to be included in the input state. High values incorporate validation metrics from old policy updates/network states.",
    )

    ###
    parser.add_argument(
        "--state_history_d",
        default=1,
        type=int,
        help="History of sampling distribution values to be included into the state. Default only includes previous parameters.",
    )

    ###
    parser.add_argument(
        "--policy_init_distr",
        default="uniform_low",
        type=str,
        help="Type of initial distribution to use. Default is uniform_low, which places high probabilities between [0.3 and 0.7]. Other options are: random, uniform_high, uniform_avg, uniform_low, uniform_low_and_tight, uniform_lowish_and_tight and the respective normal variants. You may also set uniform/low and set the mean/std in --policy_init_params.",
    )
    parser.add_argument(
        "--policy_init_params",
        nargs="+",
        default=[0.5, 0.04],
        type=float,
        help="Custom initial distribution parameters for either normal or uniform initial distributions: e.g. [mu, sig] = [0.5, 0.04] for normal.",
    )

    parser.add_argument(
        "--policy_merge_oobs",
        action="store_true",
        help="Self-Regularisation Pt.1: Values below lower interval bound are controlled together (same sampling bin).",
    )
    parser.add_argument(
        "--policy_include_pos",
        action="store_true",
        help="Self-Regularisation Pt.2: Include positives into negative sample selection. Excludes positive==anchor.",
    )
    parser.add_argument(
        "--policy_include_same",
        action="store_true",
        help="Self-Regularisation Pt.3: Specifically include positive==anchor in negatives as well.",
    )

    parser.add_argument("--ppo-ratio", default=0.2, type=float, help="ε in paper")
    parser.add_argument("--ppo-gamma", default=0.99, type=float, help="γ in advantage calculation")
    parser.add_argument("--gae-lambda", default=0.95, type=float, help="λ in GAE calculation")

    parser.add_argument("--horizon", default=6, type=int, help="T in paper")
    parser.add_argument("--policy-n-epochs", default=1, type=int, help="num. epochs of policy")
    parser.add_argument("--warmup", default=0, type=int, help="num. warmup epochs of policy")
    parser.add_argument("--resume", type=bool, default=False, help="resume training or not")

    args = parser.parse_args()

    # mods
    args.batch_size = 128
    args.n_epochs = 100
    args.policy_state_metrics = ["recall", "map", "dists"]
    args.policy_reward_metrics = ["map"]
    # args.tau = [20]
    # args.gae_lambda = 1.0
    args.horizon = 10
    args.policy_n_epochs = 3
    # args.n_bins = 20
    # args.warmup = 20
    # args.resume = True
    # args.policy_action_space = [-1, 0, 1]
    # args.policy_init_distr = "random"

    return args
