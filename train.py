"""Raccourci : ``python train.py [options]`` ≡ ``python main.py train [options]``."""

import sys

from main import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
