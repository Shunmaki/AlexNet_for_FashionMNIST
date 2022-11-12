#!/usr/bin/env python
# coding: utf-8
import argparse

def main(arg):
    if arg.train:
        import mylib.training.py
    elif arg.test:
        import mylib.test.py

def parse_input():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('-g', '--train', default=False, help='training AlexNet and save parameters')
    parser.add_argument('-i', '--test', default=True, help='infer test data and show accuracy')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(args)