import tensorflow as tf
import glob
import sys

sess = tf.Session()

if len(sys.argv) < 2:
    print("USAGE: import_and_inspect.py checkpoint_name")
    exit(-1)

saver = tf.train.Saver();

sess.__enter__()
saver.restore(sess, sys.argv[1])
print("Restored. Call this script with python -i to interact with it.")