import argparse
from collections import OrderedDict

import torch

def convert(src, dst):
    """Convert keys in DynPerceiver pretrained RegNet models to mmdet style."""
    # load caffe model
    regnet_model = torch.load(src)
    blobs = regnet_model['model']
    # convert to pytorch style
    state_dict = OrderedDict()
    for key, weight in blobs.items():
        state_dict["dyn_perceiver." + key] = weight
        print("dyn_perceiver." + key)
    
    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
