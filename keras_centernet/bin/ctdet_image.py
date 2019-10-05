#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import keras

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
set_session(session)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', default='assets/demo.jpg', type=str)
  parser.add_argument('--output', default='output', type=str)
  parser.add_argument('--inres', default='512,512', type=str)
  args, _ = parser.parse_known_args()
  args.inres = tuple(int(x) for x in args.inres.split(','))
  os.makedirs(args.output, exist_ok=True)
  kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'weights': 'ctdet_coco',
    'inres': args.inres,
  }
  heads = {
    'hm': 80,  # 3
    'reg': 2,  # 4
    'wh': 2  # 5
  }
  model = HourglassNetwork(heads=heads, **kwargs)
  model = CtDetDecode(model)
  drawer = COCODrawer()
  fns = sorted(glob(args.fn))
  for fn in tqdm(fns):
    img = cv2.imread(fn)
    letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = model.predict(pimg)[0]
    for d in detections:
      x1, y1, x2, y2, score, cl = d
      if score < 0.3:
        break
      x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
      img = drawer.draw_box(img, x1, y1, x2, y2, cl)

    out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(fn))
    cv2.imwrite(out_fn, img)
    print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
  main()
