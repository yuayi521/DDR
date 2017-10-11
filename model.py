import tensorflow as tf
import numpy as np
from nets import resnet_v1
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 320, '')
FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    """
    define the model, we use slim's implemention of resnet
    """
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            f_cls = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # firstly, using a tanh to limit the regression range from -1 to 1, then multiplying image's size
            f_reg = slim.conv2d(g[3], 8, 1, activation_fn=tf.nn.tanh, normalizer_fn=None) * FLAGS.text_scale
    return f_cls, f_reg


def cls_loss(y_true_cls, y_pred_cls, training_mask):

    y_true_cls = y_true_cls * training_mask

    y_true_cls = tf.cast(y_true_cls, tf.float32)
    y_pred_cls = tf.cast(y_pred_cls, tf.float32)

    exper_1 = tf.sign(0.5 - y_true_cls)
    exper_2 = y_pred_cls - y_true_cls
    s_square = tf.cast(tf.shape(y_true_cls)[0] * tf.shape(y_true_cls)[0], tf.float32)
    loss_cls = tf.reduce_sum(tf.square(tf.maximum(0.0, exper_1 * exper_2))) / s_square
    return loss_cls


def loss(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, training_mask):

    loss_cls = cls_loss(y_true_cls, y_pred_cls, training_mask)
    # scale the classification loss to match regression loss
    loss_cls *= 0.1
    tf.summary.scalar('classification_loss', loss_cls)

    abs_val = tf.abs(y_true_reg - y_pred_reg)
    smooth = tf.where(tf.greater(1.0, abs_val),
                      0.5 * abs_val ** 2.0,
                      abs_val - 0.5)

    # expand the dimension of y_true_cls from (80, 80, 1)to (80, 80, 8)
    iter_ = y_true_cls
    for i in xrange(7):
        y_true_cls = tf.concat([y_true_cls, iter_], axis=-1)

    # expand the dimension of training_mask from (80, 80, 1)to (80, 80, 8)
    iter_ = training_mask
    for i in xrange(7):
        training_mask = tf.concat([training_mask, iter_], axis=-1)

    loss_reg = tf.where(tf.greater(y_true_cls * training_mask, 0.0),
                        smooth,
                        0.0 * smooth)
    # loss_reg = tf.reduce_sum(loss_reg)
    loss_reg = tf.reduce_mean(loss_reg)
    tf.summary.scalar('regression_loss', loss_reg)

    loss_all = loss_reg + loss_cls

    tf.summary.scalar('all_loss', loss_all)

    return loss_all


