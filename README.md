# neural-hash-collider

Find target [hash collisions] for Apple's [NeuralHash] perceptual hash function.

For example, starting from a picture of [this
cat](https://github.com/anishathalye/neural-hash-collider/raw/assets/cat.jpg),
we can find an adversarial image that has the same hash as the
[picture](https://user-images.githubusercontent.com/1328/129860794-e7eb0132-d929-4c9d-b92e-4e4faba9e849.png)
of the dog in [this post][hash collisions]:

```console
$ python collide.py --image cat.jpg --target 59a34eabe31910abfb06f308
...
# took about 2.5 minutes to run on an i7-5930K
```

![Cat image with NeuralHash 59a34eabe31910abfb06f308](https://raw.githubusercontent.com/anishathalye/neural-hash-collider/assets/cat-adv.png) ![Dog image with NeuralHash 59a34eabe31910abfb06f308](https://raw.githubusercontent.com/anishathalye/neural-hash-collider/assets/dog.png)

We can confirm the hash collision using `nnhash.py` from
[AsuharietYgvar/AppleNeuralHash2ONNX]:

```console
$ python nnhash.py dog.png
59a34eabe31910abfb06f308
$ python nnhash.py adv.png
59a34eabe31910abfb06f308
```

[hash collisions]: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX/issues/1
[NeuralHash]: https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf

## How it works

NeuralHash is a [perceptual hash
function](https://en.wikipedia.org/wiki/Perceptual_hashing) that uses a neural
network. Images are resized to 360x360 and passed through a neural network to
produce a 128-dimensional feature vector. Then, the vector is projected onto
R^96 using a 128x96 "seed" matrix. Finally, to produce a 96-bit hash, the
96-dimensional vector is thresholded: negative entries turn into a `0` bit, and
non-negative entries turn into a `1` bit.

This entire process, except for the thresholding, is differentiable, so we can
use gradient descent to find hash collisions. This is a well-known property of
neural networks, that they are vulnerable to [adversarial
examples](https://arxiv.org/abs/1312.6199).

We can define a loss that captures how close an image is to a given target
hash: this loss is basically just the NeuralHash algorithm as described above,
but with the final "hard" thresholding step tweaked so that it is "soft" (in
particular, differentiable). Exactly how this is done (choices of activation
functions, parameters, etc.) can affect convergence, so it can require some
experimentation. Refer to `collide.py` to see what the implementation currently
does.

After choosing the loss function, we can follow the standard method to find
adversarial examples for neural networks: we perform gradient descent.

## Prerequisites

- Get Apple's NeuralHash model following the instructions in
  [AsuharietYgvar/AppleNeuralHash2ONNX] and either put all the
  files in this directory or supply the `--model` / `--seed` arguments
- Install Python dependencies: `pip install -r requirements.txt`

[AsuharietYgvar/AppleNeuralHash2ONNX]: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

## Usage

Run `python collide.py --image [path to image] --target [target hash]` to
generate a hash collision. Run `python collide.py --help` to see all the
options, including some knobs you can tweak, like the learning rate and some
other parameters.

## Limitations

The code in this repository is intended to be a demonstration, and perhaps a
starting point for other exploration. Tweaking the implementation (choice of
loss function, choice of parameters, etc.) might produce much better results
than this code currently achieves.

The code in this repository currently implements a simple loss function that
just measures the distance to the target hash value. It happens to be the case
that starting from a particular image produces a final image that looks
somewhat similar; to better enforce this property, the loss function could be
modified to add a penalty for making the image look different, e.g. l2 distance
between the original image and the computed adversarial example (another
standard technique), or we could use projected gradient descent to project onto
an l-infinity ball centered at the original image as we optimize (yet another
standard technique).

The code in this repository does not currently use any fancy optimization
algorithm, just vanilla gradient descent.
