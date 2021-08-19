# neural-hash-collider

Find target [hash collisions] for Apple's [NeuralHash] perceptual hash function.

For example, starting from a picture of [this
cat](https://github.com/anishathalye/neural-hash-collider/raw/assets/cat.jpg),
we can find an adversarial image that has the same hash as the
[picture](https://user-images.githubusercontent.com/1328/129860794-e7eb0132-d929-4c9d-b92e-4e4faba9e849.png)
of the dog in [this post][hash collisions]:

```console
$ python collide.py --image cat.jpg
...
```

![Cat image with NeuralHash 59a34eabe31910abfb06f308](https://raw.githubusercontent.com/anishathalye/neural-hash-collider/assets/cat-adv.png)

We can confirm the hash collision using `nnhash.py` from
[AsuharietYgvar/AppleNeuralHash2ONNX]:

```console
$ python nnhash.py adv.png
59a34eabe31910abfb06f308
```

[hash collisions]: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX/issues/1
[NeuralHash]: https://www.apple.com/child-safety/pdf/CSAM_Detection_Technical_Summary.pdf

## Prerequisites

- Get Apple's NeuralHash model following the instructions in
  [AsuharietYgvar/AppleNeuralHash2ONNX] and either put all the
  files in this directory or supply the `--model` / `--seed` arguments
- Install Python dependencies: `pip install onnx coremltools onnx_tf tensorflow
  numpy Pillow`

[AsuharietYgvar/AppleNeuralHash2ONNX]: https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX

## Usage

Run `python collide.py --image [path to image] --target [target hash]` to
generate a hash collision. Run `python collide.py --help` to see all the
options, including some knobs you can tweak, like the learning rate and some
other parameters.
