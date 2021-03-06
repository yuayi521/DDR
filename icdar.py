import numpy as np
import glob as glob
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from keras.engine.training import GeneratorEnqueuer

tf.app.flags.DEFINE_string('training_data_path', '/data/ocr/icdar13_15/', 'training dataset to use')
tf.app.flags.DEFINE_integer('min_text_size', 10, 'if the text size is smaller than this, we ignore it during training')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files


def get_textpolys_from_text(txt_fname):
    """
    read all text polys' 4 points coordinates, sotre in list text_polys
    :param txt_fname:
    :return: text_polys storing all text polys' coordinates
    """
    text_polys = []
    with open(txt_fname, 'rb') as f:
        for line in f:
            if '\xef\xbb\xbf' in line:
                line = line.replace('\xef\xbb\xbf', '')
            line = line.split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[:8])
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return np.array(text_polys, dtype=np.float32)


def polygon_area(poly):
    """
    copy czc's code
    compute area of a polygon
    :param poly:
    :return:
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def gurantee_polys_point_clockwise(polys, (h, w)):
    """
    copy czc's code
    check so that the text poly is in the same direction(clock-wise),
    if text poly in wrong direction(anti clock-wise), reset the text poly's direction
    and also filter some invalid polygons whose areas is too small
    :param polys:
    :return:
    """
    if polys.shape[0] == 0:
        return polys
    # clip(limit) the x, y values in an array (0, w-1), (0, h-1)
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    for poly in polys:
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print 'invalid poly'
            continue
        if p_area > 0:
            print 'poly in wrong direction'
            poly = poly[[0, 3, 2, 1], :]
        validated_polys.append(poly)
    return np.array(validated_polys)


def crop_area(im, polys, crop_background, max_tries=50):
    """
    crop image randomly, if crop_background is True then crop an area that exclude text polys,
    if crop_background is False then crop an area include text polys, both try 50 times to crop image,
    if failed then return raw image and text polys
    :param max_tries:
    :param im:
    :param polys:
    :param crop_background:
    :return:
    """
    # TODO min_crop_side_ration can set as tf.app.flags
    min_crop_side_ratio = 0.1
    h, w, _ = im.shape
    h_array, w_array = np.zeros(h, dtype=np.int32), np.zeros(w, dtype=np.int32)
    # set the range of text polys' x,y coordinates in w_array and h_array is 1
    for poly in polys:
        poly = np.round(poly).astype(np.int32)
        ymin, ymax = np.min(poly[:, 1]), np.max(poly[:, 1])
        xmin, xmax = np.min(poly[:, 0]), np.max(poly[:, 0])
        h_array[ymin:ymax] = 1
        w_array[xmin:xmax] = 1
    # ensure the cropped area not across text polys
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys

    for i in xrange(max_tries):
        random_bool = False
        if random_bool:
            yy = np.random.choice(h_axis, 2)
            xx = np.random.choice(w_axis, 2)
        else:
            yy = h_axis[0], h_axis[len(h_axis) / 2]
            xx = w_axis[0], w_axis[len(w_axis) / 2]
        x_min_carea = np.min(xx)
        x_max_carea = np.max(xx)
        y_min_carea = np.min(yy)
        y_max_carea = np.max(yy)
        if x_max_carea - x_min_carea < min_crop_side_ratio * w or y_max_carea - y_min_carea < min_crop_side_ratio * h:
            # cropped area is too samll, ignore the cropped area
            continue
        txtpolycoord_in_croparea = (polys[:, :, 0] >= x_min_carea) & (polys[:, :, 0] <= x_max_carea) & \
                                   (polys[:, :, 1] >= y_min_carea) & (polys[:, :, 1] <= y_max_carea)
        poly_idx = np.where(np.sum(txtpolycoord_in_croparea, axis=1) == 4)[0]
        if poly_idx.shape[0] == 0:
            # cropped image is background
            if crop_background:
                return im[y_min_carea: y_max_carea+1, x_min_carea: x_max_carea, :], polys[poly_idx]
            else:
                continue
        # cropped area include text polys
        im = im[y_min_carea: y_max_carea+1, x_min_carea: x_max_carea, :]
        polys = polys[poly_idx]
        # update text polys' coordinates
        polys[:, :, 0] -= x_min_carea
        polys[:, :, 1] -= y_min_carea
        return im, polys
    # tried 50 times, but failed, then return the raw(uncropped) iamge and text polys
    return im, polys


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def textpolycoord_multiply_scalefactor(polys, scale_factor):
    polys *= scale_factor
    return polys


def zoom_in_out_image_randomly(im, polys, scale_factor_arr):
    s_factor = np.random.choice(scale_factor_arr)
    im_scaled = cv2.resize(im, dsize=None, fx=s_factor, fy=s_factor)

    # update text_polys's x,y coordinates using rd_scale
    # polys *= s_factor

    polys_scaled = textpolycoord_multiply_scalefactor(polys, s_factor)
    return im_scaled, polys_scaled


def generate_labels(im_size, polys):
    """
    according to text polys calculating classification label and regression label
    :param im_size:
    :param polys:
    :return:
    """

    h, w = im_size
    # auxiliary variable for caculating y_regr_label
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    y_class_label = np.zeros((h, w), dtype=np.uint8)
    y_regr_label = np.zeros((h, w, 8), dtype=np.int32)
    # mask used during traning, to ignore some "DON'T CARE AREA" areas, weight or height less than 10 pixel
    training_mask = np.ones((h, w), dtype=np.uint8)

    for poly_idx, poly in enumerate(polys):
        poly = np.round(poly).astype(np.int32)
        # shrink the poly
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]), np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        # 1) generate classification label using shrinked poly
        cv2.fillPoly(y_class_label, [shrinked_poly], 1)
        cv2.fillPoly(poly_mask, [shrinked_poly], poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # store the y,x coordinates in text polys for calculating y_regr_label
        yx_in_txtpolys = np.argwhere(poly_mask == poly_idx + 1)
        # 2) generate regression label using raw poly
        for y, x in yx_in_txtpolys:
            y_regr_label[y, x, 0] = poly[0, 0] - x
            y_regr_label[y, x, 1] = poly[0, 1] - y
            y_regr_label[y, x, 2] = poly[1, 0] - x
            y_regr_label[y, x, 3] = poly[1, 1] - y
            y_regr_label[y, x, 4] = poly[2, 0] - x
            y_regr_label[y, x, 5] = poly[2, 1] - y
            y_regr_label[y, x, 6] = poly[3, 0] - x
            y_regr_label[y, x, 7] = poly[3, 1] - y
    return y_class_label, y_regr_label, training_mask


# def generator(input_size=320, batch_size=32, background_ration=3./8, scale_factor=np.array([0.5, 1, 2, 3]), vis=False):
def generator(input_size=320, batch_size=32, background_ration=0, scale_factor=np.array([0.5]), vis=False):

    image_arr = np.array(get_images())
    index = np.arange(0, image_arr.shape[0])
    print 'number of training images: {}'.format(len(index))
    while True:
        # np.random.shuffle(index)
        images = []
        y_class_labels = []
        y_regr_labels = []
        training_masks = []
        for i in index:
            img_file_name = image_arr[i]
            img = cv2.imread(img_file_name)
            h, w, _ = img.shape
            txt_file_name = img_file_name.replace(os.path.basename(img_file_name).split('.')[-1], 'txt')
            if not os.path.exists(txt_file_name):
                continue
            text_polys = gurantee_polys_point_clockwise(get_textpolys_from_text(txt_file_name), (h, w))

            # 2 steps(random scale, random crop) image preprocessing
            im, text_polys = zoom_in_out_image_randomly(img, text_polys, scale_factor)
            # 2) crop image randomly
            # if np.random.rand() < background_ration:
            if False:
                # cropped area is background
                im, text_polys = crop_area(im, text_polys, crop_background=True)
                if text_polys.shape[0] > 0:
                    # ensure the cropped area is background using text_polys's length
                    continue
                # pad iamge and resize iamge to input_size
                h_crop, w_crop, _ = im.shape
                max_h_w_i = np.max([h_crop, w_crop, input_size])
                padded_im = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                padded_im[:h_crop, :w_crop, :] = im.copy()
                im = cv2.resize(padded_im, dsize=(input_size, input_size))
                # set the classification label and regression label
                y_class_label = np.zeros((input_size, input_size), dtype=np.uint8)
                y_regr_lable = np.zeros((input_size, input_size, 8), dtype=np.uint8)
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            else:
                # cropped area include text poly
                im, text_polys = crop_area(im, text_polys, crop_background=False)
                if text_polys.shape[0] == 0:
                    # ensure the cropped area including text polys
                    continue
                # pad iamge and resize iamge to input_size
                h_crop, w_crop, _ = im.shape
                max_h_w_i = np.max([h_crop, w_crop, input_size])
                padded_im = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                # padded_im[:h_crop, :w_crop, :] = im.copy()
                padded_im[:h_crop, :w_crop, :] = im
                h_new, w_new, _ = padded_im.shape
                im = cv2.resize(padded_im, dsize=(input_size, input_size))
                # adjust text polys coordinates
                x_axis_ration = float(input_size) / w_new
                y_axis_ration = float(input_size) / h_new
                text_polys[:, :, 0] *= x_axis_ration
                text_polys[:, :, 1] *= y_axis_ration
                # set the classification label and regression label
                y_class_label, y_regr_lable, training_mask = generate_labels((input_size, input_size), text_polys)

            if vis:
                if 0:
                    img_path = os.path.join('/data/ocr/train_data/', os.path.basename(img_file_name))
                    cv2.imwrite(img_path, im[:, :, ::1])
                fig, axs = plt.subplots(4, 2, figsize=(10, 20))
                axs[0, 0].imshow(im[:, :, ::-1])
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])

                for poly in text_polys:
                    # draw poly
                    axs[0, 0].add_artist(patches.Polygon(
                        poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    # draw poly's height and weight
                    if False:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')

                axs[0, 1].imshow(y_class_label[::, ::])
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])

                axs[1, 0].imshow(y_regr_lable[::, ::, 0])
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])

                axs[1, 1].imshow(y_regr_lable[::, ::, 1])
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])

                axs[2, 0].imshow(y_regr_lable[::, ::, 2])
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])

                axs[2, 1].imshow(y_regr_lable[::, ::, 3])
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])

                axs[3, 0].imshow(training_mask[::, ::])
                axs[3, 0].set_xticks([])
                axs[3, 0].set_yticks([])

                plt.tight_layout()
                plt.show()
                plt.close()

            images.append(im[:, :, ::-1].astype(np.float32))
            y_class_labels.append(y_class_label[::4, ::4, np.newaxis].astype(np.float32))
            y_regr_labels.append(y_regr_lable[::4, ::4, :].astype(np.float32))
            training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

            if len(images) == batch_size:
                yield images, y_class_labels, y_regr_labels, training_masks
                # set the list to empty
                images = []
                y_class_labels = []
                y_regr_labels = []
                training_masks = []


def get_batch(num_workers=10, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), pickle_safe=True)
        enqueuer.start(max_q_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    # print self.enqueuer.queue.qsize()
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':
    gen = generator(input_size=320, batch_size=200, vis=False)
    write = False
    if write:
        for ii in gen:
            images, y_cls_label, y_regr_label, masks = ii
            # write the file into disk
            np.save('/tmp/1', images)
            np.save('/tmp/2', y_cls_label)
            np.save('/tmp/3', y_regr_label)
            np.save('/tmp/4', masks)
            break
    print 'OVer ........'

    for ii in gen:
        images, y_cls_label, y_regr_label, masks = ii
        images_wri = np.load('/tmp/1.npy', 'r')
        y_cls_label_wri = np.load('/tmp/2.npy', 'r')
        y_regr_label_wri = np.load('/tmp/3.npy', 'r')
        masks_wri = np.load('/tmp/4.npy', 'r')

        if np.array_equal(images, images_wri) and np.array_equal(y_cls_label, y_cls_label_wri) and \
                np.array_equal(y_regr_label, y_regr_label_wri) and np.array_equal(masks, masks_wri):
            print '\nOK......'
        else:
            print '\nwrong.....'

        break








