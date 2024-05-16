from mmengine.logging import print_log
from logging import INFO

def cs470_debug_print(msg):
    print_log("[CS470 DEBUG] " + msg, logger="current", level=INFO)