#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np

from batch_smpl import SMPL


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 test_runner.py <path-to-SMPL-model>")
        sys.exit()

    smpl_path = os.path.realpath(sys.argv[1])
    smpl_model = SMPL(smpl_path)

    betas = tf.random_normal([1, 10], stddev=0.1)
    thetas = tf.random_normal([1, 72], stddev=0.2)

    verts, _, _ = smpl_model(betas, thetas, get_skin=True)
    verts = verts[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    result = sess.run(verts)

    dirpath = os.path.dirname(os.path.realpath(__file__))
    faces = np.load(os.path.join(dirpath, 'smpl_faces.npy'))

    outmesh_path = os.path.join(dirpath, 'smpl_tf.obj')
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



