import tensorflow as tf
import numpy as np
from nets import resnet_v1
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 320, '')
FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


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
    '''
    define the model, we use slim's implemention of resnet
    '''
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
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            # F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            # geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            # angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            # F_geometry = tf.concat([geo_map, angle_map], axis=-1)

            f_cls = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range
            # f_reg = slim.conv2d(g[3], 8, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            f_reg = slim.conv2d(g[3], 8, 1, activation_fn=tf.nn.tanh, normalizer_fn=None) * FLAGS.text_scale

    # return F_score, F_geometry
    return f_cls, f_reg


def dice_coefficient_remove_mask(y_true_cls, y_pred_cls):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    union = tf.reduce_sum(y_true_cls) + tf.reduce_sum(y_pred_cls) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def my_cls_loss(y_true_cls, y_pred_cls):
    """

    :param y_true_cls:
    :param y_pred_cls:
    :return:
    """
    y_true_cls = tf.cast(y_true_cls, tf.float32)
    y_pred_cls = tf.cast(y_pred_cls, tf.float32)

    exper_1 = tf.sign(0.5 - y_true_cls)
    exper_2 = y_pred_cls - y_true_cls
    s_square = tf.cast(tf.shape(y_true_cls)[0] * tf.shape(y_true_cls)[0], tf.float32)
    loss_cls = tf.reduce_sum(tf.square(tf.maximum(0.0, exper_1 * exper_2))) / s_square
    return loss_cls


def my_loss(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg):
    """

    :param y_pred_reg:
    :param y_true_reg:
    :param y_true_cls:
    :param y_pred_cls:
    :return:
    """

    loss_cls = my_cls_loss(y_true_cls, y_pred_cls)
    # use czc's cls loss
    # loss_cls = dice_coefficient_remove_mask(y_true_cls, y_pred_cls)
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

    loss_reg = tf.where(tf.greater(y_true_cls, 0.0),
                        smooth,
                        0.0 * smooth)
    # loss_reg = tf.reduce_sum(loss_reg)
    loss_reg = tf.reduce_mean(loss_reg)
    tf.summary.scalar('regression_loss', loss_reg)

    loss_all = loss_reg + loss_cls

    tf.summary.scalar('all_loss', loss_all)

    return loss_all


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
