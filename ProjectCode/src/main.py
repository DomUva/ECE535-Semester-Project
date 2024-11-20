import configparser
import argparse
import logging
import os
import warnings
import torch
#from mpi4py import MPI
from fl import FL


# # For MPI experiments
# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()


def main():
    # is_mpi = COMM.Get_size() != 1
    config = read_config()
    fl = FL(config)
    fl.start()


def read_config():
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--config", type=str,
                            # help="name of the config file of simulation")
    # args = arg_parser.parse_args()
    config = configparser.ConfigParser()
    config.read(r'C:\Users\Treys\.vscode\ECE535\ECE535-Semester-Project\ProjectCode\config\opp\dccae\A0_B0_AB30_label_B_test_A')
    return config


if __name__ == "__main__":
    main()
