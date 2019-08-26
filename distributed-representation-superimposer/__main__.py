#!/usr/bin/env python

from extractor import Extractor


def main():
    extractor = Extractor('../Japanese_L-12_H-768_A-12_E-30_BPE/')
    encoded = extractor.extract('天気教えて')
    print(encoded)
    print(len(encoded[0]))


if __name__ == '__main__':
    main()
