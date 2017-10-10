import cv2
import time
import os
import numpy as np
import tensorflow as tf
import model
import matplotlib.pyplot as plt
import locality_aware_nms as nms_locality

# tf.app.flags.DEFINE_string('test_data_path', '/data/ocr/icdar2015/', '')
tf.app.flags.DEFINE_string('test_data_path', '/data/ocr/ch4/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/ddr_icdar15/', '')
tf.app.flags.DEFINE_string('output_path', '/home/yuquanjie/output_ocr/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    """
    find image files in test data path
    :return: list of files found
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print 'Find {} images'.format(len(files))
    return files


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape
    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (int(resize_h / 32) - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (int(resize_w / 32) - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def restore_rectangle(xy_text, geo_map):
    """

    :param xy_text:
    :param geo_map:
    :return:
    """
    text_box = []
    for idx in xrange(len(xy_text)):
        text_box.append([xy_text[idx][0] + geo_map[idx][0], xy_text[idx][1] + geo_map[idx][1],
                         xy_text[idx][0] + geo_map[idx][2], xy_text[idx][1] + geo_map[idx][3],
                         xy_text[idx][0] + geo_map[idx][4], xy_text[idx][1] + geo_map[idx][5],
                         xy_text[idx][0] + geo_map[idx][6], xy_text[idx][1] + geo_map[idx][7]])
    return np.array(text_box, dtype=np.float32)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2, vis=False):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :param vis:
    :return:
    """
    # reduce dimension of score map and geo_map
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # visualize the score map
    if vis:
        plt.imshow(score_map)
        plt.show()
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore text boxes from score_map and geo_map
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print '{} text boxes before nms'.format(text_box_restored.shape[0])

    # boxes = text_box_restored
    # score_map greater than threshold(0.8)
    # points = xy_text[:, ::-1] * 4

    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start

    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer
        # return None, None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) / 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    # return boxes, points.astype(np.float32), timer
    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print 'Restore from {}'.format(model_path)
            saver.restore(sess, model_path)
            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print '====>{}'.format(im_fn)
                im = cv2.imread(im_fn)[:, :, ::-1]
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                # boxes, points, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                print '{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000)
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                # save to file
                if boxes is not None:
                    with open(
                            os.path.join(FLAGS.output_path, 'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0])),
                            'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(255, 255, 0), thickness=1)
                            # break

                        # points[:, 0] /= ratio_w
                        # points[:, 1] /= ratio_h
                        # for point in points:
                        #     cv2.circle(im[:, :, ::-1], (point[0], point[1]), 1, color=(255, 0, 0))
                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), im[:, :, ::-1])


if __name__ == '__main__':
    tf.app.run()
