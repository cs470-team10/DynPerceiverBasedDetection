from mmengine.logging import print_log
from logging import INFO

def cs470_print(msg):
    print_log("[CS470 LOG] " + msg, logger="current", level=INFO)