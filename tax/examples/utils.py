import argparse


def parse_args() -> argparse.ArgumentParser:
    """Get terminal arguments

    Returns:
        argparse.ArgumentParser: passed arguments
    """
    # construct the argument parser and parser the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration",
    )
    parser.add_argument(
        "--base_dir",
        default="/mnt/c/Users/Rares/Documents/phd/diffusion_models/diffusion/data/",
        type=str,
        help="directory where to dump training output",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Flag to mark evaluation or training",
    )
    args = parser.parse_args()
    return args
