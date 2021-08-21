# Copyright (c) 2021 Anish Athalye. Released under the MIT license.

import numpy as np
import tensorflow as tf
import argparse
import os

from util import *


DEFAULT_MODEL_PATH = 'model.onnx'
DEFAULT_SEED_PATH = 'neuralhash_128x96_seed1.dat'
DEFAULT_TARGET_HASH = '59a34eabe31910abfb06f308'
DEFAULT_ITERATIONS = 1000
DEFAULT_LR = 2.0
DEFAULT_K = 10.0
DEFAULT_CLIP_RANGE = 0.1
DEFAULT_SAVE_ITERATIONS = 100


def main():
    tf.compat.v1.disable_eager_execution()
    options = get_options()

    model = load_model(options.model)
    image = model.tensor_dict['image']
    logits = model.tensor_dict['leaf/logits']
    seed = load_seed(options.seed)

    target = hash_from_hex(options.target)

    x = load_image(options.image)
    h = hash_from_hex(options.target)

    with model.graph.as_default():
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            proj = tf.reshape(tf.linalg.matmul(seed, tf.reshape(logits, (128, 1))), (96,))
            normalized, _ = tf.linalg.normalize(proj)
            hash_output = tf.sigmoid(normalized * options.k)

            hash_output = tf.clip_by_value(hash_output, options.clip_range, 1.0 - options.clip_range) - 0.5
            hash_output = hash_output * (0.5 / (0.5 - options.clip_range))
            hash_output = hash_output + 0.5

            loss = tf.math.reduce_sum(tf.math.squared_difference(hash_output, h))
            g, = tf.gradients(loss, image)

            best = float('inf')

            for i in range(options.iterations):
                xq = quantize(x)
                hash_output_v, loss_v, g_v = sess.run([hash_output, loss, g], feed_dict={image: xq})
                dist = np.sum((hash_output_v >= 0.5) != (h >= 0.5))
                if dist < best or i % options.save_iterations == 0:
                    save_image(x, os.path.join(options.save_directory, 'out_iter={:05d}_dist={:02d}.png'.format(i, dist)))
                if dist < best:
                    best = dist
                g_v_norm = g_v / np.linalg.norm(g_v)
                x = x - options.learning_rate * g_v_norm
                x = x.clip(-1, 1)
                print('iteration: {}/{}, best: {}, hash: {}, distance: {}, loss: {:.3f}'.format(
                    i+1,
                    options.iterations,
                    best,
                    hash_to_hex(hash_output_v),
                    dist,
                    loss_v
                ))


def quantize(x):
    x = (x + 1.0) * (255.0 / 2.0)
    x = x.astype(np.uint8).astype(np.float32)
    x = x / (255.0 / 2.0) - 1.0
    return x


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='path to starting image', required=True)
    parser.add_argument('--model', help='path to model', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--seed', help='path to seed', default=DEFAULT_SEED_PATH)
    parser.add_argument('--target', help='target hash', default=DEFAULT_TARGET_HASH)
    parser.add_argument('--learning-rate', type=float, help='learning rate', default=DEFAULT_LR)
    parser.add_argument('--k', type=float, help='k parameter', default=DEFAULT_K)
    parser.add_argument('--clip-range', type=float, help='clip range parameter', default=DEFAULT_CLIP_RANGE)
    parser.add_argument('--iterations', type=int, help='max number of iterations', default=DEFAULT_ITERATIONS)
    parser.add_argument('--save-directory', help='directory to save output images', default='.')
    parser.add_argument('--save-iterations', type=int, help='save every _ iterations, regardless of improvement', default=DEFAULT_SAVE_ITERATIONS)
    return parser.parse_args()


if __name__ == '__main__':
    main()
