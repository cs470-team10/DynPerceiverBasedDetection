import argparse
from collections import OrderedDict

import torch

def convert(src, dst):
    """Convert keys in Pytorch RegNet models to mmdet style."""
    regnet_model = torch.load(src)
    blobs = regnet_model
    # convert to pytorch style
    state_dict = OrderedDict()
    for key, weight in blobs.items():
        if "num_batches_tracked" in key:
            continue
        if ("fc.weight" in key) or ("fc.bias" in key):
            continue
        new_key = ""
        if ("stem" in key):
            new_key = "cnn_" + key
        else:
            new_key = key.replace("trunk_output.", "cnn_body.")
        state_dict[new_key] = weight
        print(new_key)
    
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

# regnet.cnn_stem.0.weight
# regnet.cnn_stem.1.weight
# regnet.cnn_stem.1.bias
# regnet.cnn_body.block1.block1-0.proj.0.weight
# regnet.cnn_body.block1.block1-0.proj.1.weight
# regnet.cnn_body.block1.block1-0.proj.1.bias
# regnet.cnn_body.block1.block1-0.f.a.0.weight
# regnet.cnn_body.block1.block1-0.f.a.1.weight
# regnet.cnn_body.block1.block1-0.f.a.1.bias
# regnet.cnn_body.block1.block1-0.f.b.0.weight
# regnet.cnn_body.block1.block1-0.f.b.1.weight
# regnet.cnn_body.block1.block1-0.f.b.1.bias
# regnet.cnn_body.block1.block1-0.f.se.fc1.weight
# regnet.cnn_body.block1.block1-0.f.se.fc1.bias
# regnet.cnn_body.block1.block1-0.f.se.fc2.weight
# regnet.cnn_body.block1.block1-0.f.se.fc2.bias
# regnet.cnn_body.block1.block1-0.f.c.0.weight
# regnet.cnn_body.block1.block1-0.f.c.1.weight
# regnet.cnn_body.block1.block1-0.f.c.1.bias
# regnet.cnn_body.block2.block2-0.proj.0.weight
# regnet.cnn_body.block2.block2-0.proj.1.weight
# regnet.cnn_body.block2.block2-0.proj.1.bias
# regnet.cnn_body.block2.block2-0.f.a.0.weight
# regnet.cnn_body.block2.block2-0.f.a.1.weight
# regnet.cnn_body.block2.block2-0.f.a.1.bias
# regnet.cnn_body.block2.block2-0.f.b.0.weight
# regnet.cnn_body.block2.block2-0.f.b.1.weight
# regnet.cnn_body.block2.block2-0.f.b.1.bias
# regnet.cnn_body.block2.block2-0.f.se.fc1.weight
# regnet.cnn_body.block2.block2-0.f.se.fc1.bias
# regnet.cnn_body.block2.block2-0.f.se.fc2.weight
# regnet.cnn_body.block2.block2-0.f.se.fc2.bias
# regnet.cnn_body.block2.block2-0.f.c.0.weight
# regnet.cnn_body.block2.block2-0.f.c.1.weight
# regnet.cnn_body.block2.block2-0.f.c.1.bias
# regnet.cnn_body.block2.block2-1.f.a.0.weight
# regnet.cnn_body.block2.block2-1.f.a.1.weight
# regnet.cnn_body.block2.block2-1.f.a.1.bias
# regnet.cnn_body.block2.block2-1.f.b.0.weight
# regnet.cnn_body.block2.block2-1.f.b.1.weight
# regnet.cnn_body.block2.block2-1.f.b.1.bias
# regnet.cnn_body.block2.block2-1.f.se.fc1.weight
# regnet.cnn_body.block2.block2-1.f.se.fc1.bias
# regnet.cnn_body.block2.block2-1.f.se.fc2.weight
# regnet.cnn_body.block2.block2-1.f.se.fc2.bias
# regnet.cnn_body.block2.block2-1.f.c.0.weight
# regnet.cnn_body.block2.block2-1.f.c.1.weight
# regnet.cnn_body.block2.block2-1.f.c.1.bias
# regnet.cnn_body.block2.block2-2.f.a.0.weight
# regnet.cnn_body.block2.block2-2.f.a.1.weight
# regnet.cnn_body.block2.block2-2.f.a.1.bias
# regnet.cnn_body.block2.block2-2.f.b.0.weight
# regnet.cnn_body.block2.block2-2.f.b.1.weight
# regnet.cnn_body.block2.block2-2.f.b.1.bias
# regnet.cnn_body.block2.block2-2.f.se.fc1.weight
# regnet.cnn_body.block2.block2-2.f.se.fc1.bias
# regnet.cnn_body.block2.block2-2.f.se.fc2.weight
# regnet.cnn_body.block2.block2-2.f.se.fc2.bias
# regnet.cnn_body.block2.block2-2.f.c.0.weight
# regnet.cnn_body.block2.block2-2.f.c.1.weight
# regnet.cnn_body.block2.block2-2.f.c.1.bias
# regnet.cnn_body.block3.block3-0.proj.0.weight
# regnet.cnn_body.block3.block3-0.proj.1.weight
# regnet.cnn_body.block3.block3-0.proj.1.bias
# regnet.cnn_body.block3.block3-0.f.a.0.weight
# regnet.cnn_body.block3.block3-0.f.a.1.weight
# regnet.cnn_body.block3.block3-0.f.a.1.bias
# regnet.cnn_body.block3.block3-0.f.b.0.weight
# regnet.cnn_body.block3.block3-0.f.b.1.weight
# regnet.cnn_body.block3.block3-0.f.b.1.bias
# regnet.cnn_body.block3.block3-0.f.se.fc1.weight
# regnet.cnn_body.block3.block3-0.f.se.fc1.bias
# regnet.cnn_body.block3.block3-0.f.se.fc2.weight
# regnet.cnn_body.block3.block3-0.f.se.fc2.bias
# regnet.cnn_body.block3.block3-0.f.c.0.weight
# regnet.cnn_body.block3.block3-0.f.c.1.weight
# regnet.cnn_body.block3.block3-0.f.c.1.bias
# regnet.cnn_body.block3.block3-1.f.a.0.weight
# regnet.cnn_body.block3.block3-1.f.a.1.weight
# regnet.cnn_body.block3.block3-1.f.a.1.bias
# regnet.cnn_body.block3.block3-1.f.b.0.weight
# regnet.cnn_body.block3.block3-1.f.b.1.weight
# regnet.cnn_body.block3.block3-1.f.b.1.bias
# regnet.cnn_body.block3.block3-1.f.se.fc1.weight
# regnet.cnn_body.block3.block3-1.f.se.fc1.bias
# regnet.cnn_body.block3.block3-1.f.se.fc2.weight
# regnet.cnn_body.block3.block3-1.f.se.fc2.bias
# regnet.cnn_body.block3.block3-1.f.c.0.weight
# regnet.cnn_body.block3.block3-1.f.c.1.weight
# regnet.cnn_body.block3.block3-1.f.c.1.bias
# regnet.cnn_body.block3.block3-2.f.a.0.weight
# regnet.cnn_body.block3.block3-2.f.a.1.weight
# regnet.cnn_body.block3.block3-2.f.a.1.bias
# regnet.cnn_body.block3.block3-2.f.b.0.weight
# regnet.cnn_body.block3.block3-2.f.b.1.weight
# regnet.cnn_body.block3.block3-2.f.b.1.bias
# regnet.cnn_body.block3.block3-2.f.se.fc1.weight
# regnet.cnn_body.block3.block3-2.f.se.fc1.bias
# regnet.cnn_body.block3.block3-2.f.se.fc2.weight
# regnet.cnn_body.block3.block3-2.f.se.fc2.bias
# regnet.cnn_body.block3.block3-2.f.c.0.weight
# regnet.cnn_body.block3.block3-2.f.c.1.weight
# regnet.cnn_body.block3.block3-2.f.c.1.bias
# regnet.cnn_body.block3.block3-3.f.a.0.weight
# regnet.cnn_body.block3.block3-3.f.a.1.weight
# regnet.cnn_body.block3.block3-3.f.a.1.bias
# regnet.cnn_body.block3.block3-3.f.b.0.weight
# regnet.cnn_body.block3.block3-3.f.b.1.weight
# regnet.cnn_body.block3.block3-3.f.b.1.bias
# regnet.cnn_body.block3.block3-3.f.se.fc1.weight
# regnet.cnn_body.block3.block3-3.f.se.fc1.bias
# regnet.cnn_body.block3.block3-3.f.se.fc2.weight
# regnet.cnn_body.block3.block3-3.f.se.fc2.bias
# regnet.cnn_body.block3.block3-3.f.c.0.weight
# regnet.cnn_body.block3.block3-3.f.c.1.weight
# regnet.cnn_body.block3.block3-3.f.c.1.bias
# regnet.cnn_body.block3.block3-4.f.a.0.weight
# regnet.cnn_body.block3.block3-4.f.a.1.weight
# regnet.cnn_body.block3.block3-4.f.a.1.bias
# regnet.cnn_body.block3.block3-4.f.b.0.weight
# regnet.cnn_body.block3.block3-4.f.b.1.weight
# regnet.cnn_body.block3.block3-4.f.b.1.bias
# regnet.cnn_body.block3.block3-4.f.se.fc1.weight
# regnet.cnn_body.block3.block3-4.f.se.fc1.bias
# regnet.cnn_body.block3.block3-4.f.se.fc2.weight
# regnet.cnn_body.block3.block3-4.f.se.fc2.bias
# regnet.cnn_body.block3.block3-4.f.c.0.weight
# regnet.cnn_body.block3.block3-4.f.c.1.weight
# regnet.cnn_body.block3.block3-4.f.c.1.bias
# regnet.cnn_body.block3.block3-5.f.a.0.weight
# regnet.cnn_body.block3.block3-5.f.a.1.weight
# regnet.cnn_body.block3.block3-5.f.a.1.bias
# regnet.cnn_body.block3.block3-5.f.b.0.weight
# regnet.cnn_body.block3.block3-5.f.b.1.weight
# regnet.cnn_body.block3.block3-5.f.b.1.bias
# regnet.cnn_body.block3.block3-5.f.se.fc1.weight
# regnet.cnn_body.block3.block3-5.f.se.fc1.bias
# regnet.cnn_body.block3.block3-5.f.se.fc2.weight
# regnet.cnn_body.block3.block3-5.f.se.fc2.bias
# regnet.cnn_body.block3.block3-5.f.c.0.weight
# regnet.cnn_body.block3.block3-5.f.c.1.weight
# regnet.cnn_body.block3.block3-5.f.c.1.bias
# regnet.cnn_body.block3.block3-6.f.a.0.weight
# regnet.cnn_body.block3.block3-6.f.a.1.weight
# regnet.cnn_body.block3.block3-6.f.a.1.bias
# regnet.cnn_body.block3.block3-6.f.b.0.weight
# regnet.cnn_body.block3.block3-6.f.b.1.weight
# regnet.cnn_body.block3.block3-6.f.b.1.bias
# regnet.cnn_body.block3.block3-6.f.se.fc1.weight
# regnet.cnn_body.block3.block3-6.f.se.fc1.bias
# regnet.cnn_body.block3.block3-6.f.se.fc2.weight
# regnet.cnn_body.block3.block3-6.f.se.fc2.bias
# regnet.cnn_body.block3.block3-6.f.c.0.weight
# regnet.cnn_body.block3.block3-6.f.c.1.weight
# regnet.cnn_body.block3.block3-6.f.c.1.bias
# regnet.cnn_body.block3.block3-7.f.a.0.weight
# regnet.cnn_body.block3.block3-7.f.a.1.weight
# regnet.cnn_body.block3.block3-7.f.a.1.bias
# regnet.cnn_body.block3.block3-7.f.b.0.weight
# regnet.cnn_body.block3.block3-7.f.b.1.weight
# regnet.cnn_body.block3.block3-7.f.b.1.bias
# regnet.cnn_body.block3.block3-7.f.se.fc1.weight
# regnet.cnn_body.block3.block3-7.f.se.fc1.bias
# regnet.cnn_body.block3.block3-7.f.se.fc2.weight
# regnet.cnn_body.block3.block3-7.f.se.fc2.bias
# regnet.cnn_body.block3.block3-7.f.c.0.weight
# regnet.cnn_body.block3.block3-7.f.c.1.weight
# regnet.cnn_body.block3.block3-7.f.c.1.bias
# regnet.cnn_body.block4.block4-0.proj.0.weight
# regnet.cnn_body.block4.block4-0.proj.1.weight
# regnet.cnn_body.block4.block4-0.proj.1.bias
# regnet.cnn_body.block4.block4-0.f.a.0.weight
# regnet.cnn_body.block4.block4-0.f.a.1.weight
# regnet.cnn_body.block4.block4-0.f.a.1.bias
# regnet.cnn_body.block4.block4-0.f.b.0.weight
# regnet.cnn_body.block4.block4-0.f.b.1.weight
# regnet.cnn_body.block4.block4-0.f.b.1.bias
# regnet.cnn_body.block4.block4-0.f.se.fc1.weight
# regnet.cnn_body.block4.block4-0.f.se.fc1.bias
# regnet.cnn_body.block4.block4-0.f.se.fc2.weight
# regnet.cnn_body.block4.block4-0.f.se.fc2.bias
# regnet.cnn_body.block4.block4-0.f.c.0.weight
# regnet.cnn_body.block4.block4-0.f.c.1.weight
# regnet.cnn_body.block4.block4-0.f.c.1.bias
# regnet.cnn_body.block4.block4-1.f.a.0.weight
# regnet.cnn_body.block4.block4-1.f.a.1.weight
# regnet.cnn_body.block4.block4-1.f.a.1.bias
# regnet.cnn_body.block4.block4-1.f.b.0.weight
# regnet.cnn_body.block4.block4-1.f.b.1.weight
# regnet.cnn_body.block4.block4-1.f.b.1.bias
# regnet.cnn_body.block4.block4-1.f.se.fc1.weight
# regnet.cnn_body.block4.block4-1.f.se.fc1.bias
# regnet.cnn_body.block4.block4-1.f.se.fc2.weight
# regnet.cnn_body.block4.block4-1.f.se.fc2.bias
# regnet.cnn_body.block4.block4-1.f.c.0.weight
# regnet.cnn_body.block4.block4-1.f.c.1.weight
# regnet.cnn_body.block4.block4-1.f.c.1.bias