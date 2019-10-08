#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os

from tensorflow import keras

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda
from tqdm import tqdm
from glob import glob
import tensorflow as tf

import tensorflow.keras.backend as K

from keras_centernet.models.decode import ctdet_decode_internal
from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode
from keras_centernet.utils.utils import COCODrawer
from keras_centernet.utils.letterbox import LetterboxTransformer
from tensorflow.keras.backend import set_session
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', default='assets/demo.jpg', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--inres', default='512,512', type=str)
    args, _ = parser.parse_known_args()
    args.inres = tuple(int(x) for x in args.inres.split(','))
    # os.makedirs(args.output, exist_ok=True)
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

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config=config)
    set_session(session)

    def get_name(x, prefix):
        return prefix + x.name.split("/")[0]

    def _recover_from_clipped(x, v_min, v_max, max_u8=255):
        x = tf.cast(x, tf.float32)
        return x / max_u8 * (v_max - v_min) + v_min

    def _clip_to_uint8(x, v_min, v_max, max_u8=255.):
        x = tf.maximum(x, v_min)
        x = tf.minimum(x, v_max)
        x = (x - v_min) / (v_max * 1.0 - v_min)
        x = tf.cast(x * max_u8, tf.uint8)
        return x

    class Clip(Layer):
        def __init__(self, vmin=0, vmax=10, *argv, **kwargs):
            super(Clip, self).__init__(*argv,**kwargs)
            self.vmin = vmin
            self.vmax = vmax
        def call(self, x):
            return _clip_to_uint8(x, self.vmin, self.vmax)
        def get_config(self):
            config = {
                "vmin":self.vmin
                , "vmax":self.vmax
            }
            base_config = super(Clip, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    def uint8_wrapper(model, output_max):
        inres = K.int_shape(model.inputs[0])
        input = Input(shape=(inres[1], inres[2], 3), dtype="uint8", name='Input')
        def normalize(x):
            mean = K.constant([0.40789655, 0.44719303, 0.47026116])
            std = K.constant([0.2886383, 0.27408165, 0.27809834])
            return (K.cast(x,'float32') - mean) / std
        x = Lambda(normalize)(input)
        outputs = model(x)

        output_u8 = [Clip(vmin, vmax,name=name)(output)
                     for (output, (vmin, vmax), name) in zip(outputs, output_max
                                                             , [get_name(x, "u8_") for x in model.outputs])
                     ]
        return Model(input, output_u8)

    hourglass = HourglassNetwork(heads=heads, **kwargs)
    mx = max(args.inres)
    inv_scale = 4
    output_max = [(-10, 10), (0, inv_scale), (0, mx // inv_scale), (-10, 10), (0, inv_scale), (0, mx // inv_scale)]

    uint8_model = uint8_wrapper(hourglass, output_max)
    uint8_model.predict(np.ones((1, 512, 512, 3), dtype=np.uint8))
    uint8_model.save('keras')

    def representative_dataset_gen(num_calibration_steps=1):
        for _ in range(num_calibration_steps):
            # Get sample input data as a numpy array in a method of your choosing.
            yield [np.ones(1, 512, 512, 3, dtype=np.uint8)]
    #
    # session.run(tf.global_variables_initializer())
    # tf.compat.v1.saved_model.simple_save(session, "/keras-centernet/tfmodel",
    #                                      {"inputs": uint8_model.inputs[0]}
    #                                      , dict(list(zip(["a", "b", "c", "d", "e", "f"], uint8_model.outputs))))
    # saved_model_dir, input_arrays=None, input_shapes=None, output_arrays=None, tag_set=None, signature_key=None
    #function(model_file, input_arrays=None, input_shapes=None, output_arrays=None, custom_objects=None)
    converter = tf.lite.TFLiteConverter.from_keras_model_file('keras', ["KInput"], input_shapes={"UInput": [1, 512, 512, 3]},custom_objects={"Clip":Clip})

    # converter = tf.lite.TFLiteConverter.from_saved_model("/keras-centernet/tfmodel", uint8_model.inputs,
    #                                                      {"UInput": [1, 512, 512, 3]},
    #                                                      uint8_model.outputs)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    # converter = tf.lite.TFLiteConverter.from_session(session, uint8_model.inputs, uint8_model.outputs)
    converter.representative_dataset = representative_dataset_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # drawer = COCODrawer()
    # fns = sorted(glob(args.fn))
    # for fn in tqdm(fns):
    #   img = cv2.imread(fn)
    #   letterbox_transformer = LetterboxTransformer(args.inres[0], args.inres[1])
    #   pimg = letterbox_transformer(img)
    #   pimg = normalize_image(pimg)
    #   pimg = np.expand_dims(pimg, 0)
    #   detections = model.predict(pimg)[0]
    #   for d in detections:
    #     x1, y1, x2, y2, score, cl = d
    #     if score < 0.3:
    #       break
    #     x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
    #     img = drawer.draw_box(img, x1, y1, x2, y2, cl)
    #
    #   out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(fn))
    #   cv2.imwrite(out_fn, img)
    #   print("Image saved to: %s" % out_fn)


if __name__ == '__main__':
    main()
