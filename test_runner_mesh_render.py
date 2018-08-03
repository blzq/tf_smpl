#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .batch_smpl import SMPL
from ..tf_mesh_renderer import mesh_renderer


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
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
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

    in_verts = tf.constant(result, dtype=tf.float32)[tf.newaxis]
    in_normals = tf.nn.l2_normalize(in_verts, axis=1)
    faces = tf.constant(faces, dtype=tf.int32)

    eye = tf.constant([[0.0, 0.0, 6.0]], dtype=tf.float32)
    center = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    world_up = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
    image_width, image_height = 640, 480
    light_positions = tf.constant([[[0.0, 0.0, 6.0]]])
    light_intensities = tf.zeros([1, 1, 3], dtype=tf.float32)
    vertex_diffuse_colors = tf.ones_like(in_verts, dtype=tf.float32)

    rendered = mesh_renderer.mesh_renderer(
        in_verts, faces, in_normals,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)

    img = sess.run(rendered)
    print(img.shape)
    plt.imshow(img[0][:, :, 3])
    plt.show()
