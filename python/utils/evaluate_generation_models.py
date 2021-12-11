import argparse
import csv
import cv2 as cv2
import numpy as np
import os as os
import pandas as pd
import re as re
import utils as utils
from utils import plot_together
from math import floor

def graph(values, title, eval_folder, axes, colors, labels, legend, normalized):
    graph_str_out = eval_folder + title + '.png'
    plot_together(values, colors, labels, title, axes, graph_str_out, legend, normalized)
    print('Output to {}'.format(graph_str_out))

def evaluate_pretraining():
    # Initalize application
    eval_folder = '../attngan/output/'
    colors = ['mediumaquamarine', 'seagreen', 'teal', 'darkslateblue']
    paths = []
    labels = []
    for i in range(0, opt.evalRange):
        if opt.evalRange == 1:
            dim = opt.blockDim
        else:
            dim = 2**(6 - i)
        paths.append('{}/{}/{}/pre_train.csv'.format(opt.inputFolder, opt.inputSpecifier, dim))
        labels.append('{:d}'.format(dim))
    data = []

    for path in paths:
        with open(eval_folder + path) as f:
            reader = csv.reader(f)
            data.append( list(reader) )

    # Collect loss data for generator only
    values = []
    for i in range(0, len(data)):
        for loss_data in data[i]:
            losses = []
            for loss in loss_data:
                losses.append( re.sub('[^0-9]', '', loss) )

            if len(values) <= i:
                values.append([])
            values[i].append(np.average( np.array([float(losses[2]), float(losses[3])], dtype=float)))

    for i in range(0, len(values)):
        values[i] = values[i][0:(opt.numEpochs)]

    # Normalize losses
    max_loss = np.matrix(values).max()
    for i in range(0, len(values)):
        values[i] = (np.array(values[i]) / float(max_loss)).tolist()

    out_path = '../eval/generation/'
    axes = ['Epoch', 'Normalized Loss']
    graph(values, 'Pretraining Loss', out_path, axes, colors, labels, 'upper right', True)

    # Collect time data
    values = []
    for i in range(0, len(data)):
        times = []
        if len(values) <= i:
            values.append([])
        for time_data in data[int(i)]:
            times.append(float(time_data[0]))
            time = np.sum( np.array(times, dtype=float) )
            values[i].append(time / 3600.0)
        print('Pre {:d}: {:06f}'.format(2**(6 - i), np.sum(np.array(times, dtype=float)) / float(3600)))

    for i in range(0, len(values)):
        values[i] = values[i][0:(opt.numEpochs)]
    
    # Graph time data
    axes = ['Epoch', 'Time (hours)']
    graph(values, 'Training Time of Pretraining Models', out_path, axes, colors, labels, 'lower right', False)

def evaluate_gentraining():
    # Initalize application
    eval_folder = '../attngan/output/'
    colors = ['darkslategrey', 'goldenrod', 'royalblue']
    paths = []
    labels = []
    dim = opt.blockDim
    for i in range(0, 3):
        # if opt.evalRange == 1:
        #     dim = opt.blockDim
        # else:
        #     dim = 2**(6 - i)
        specifier = 'Stone'
        if i == 1:
            specifier = 'Gold'
        elif i == 2:
            specifier = 'Glass'
        paths.append('{}/{}/{}/gen_train.csv'.format(opt.inputFolder, '2f_' + specifier.lower(), dim))
        if specifier == 'Stone':
            specifier = 'Clay'
        labels.append(specifier)
    data = []

    print(paths)
    for path in paths:
        with open(eval_folder + path) as f:
            reader = csv.reader(f)
            data.append( list(reader) )

    # Collect loss data for generator only
    values = []
    for i in range(0, len(data)):
        for loss_data in data[i]:
            losses = []
            for loss in loss_data:
                losses.append( re.sub('[^0-9]', '', loss) )

            if len(values) <= i:
                values.append([])
            values[i].append(float(losses[2]))

    for i in range(0, len(values)):
        values[i] = values[i][0:(opt.numEpochs)]

    # Normalize losses
    max_loss = np.matrix(values).max()
    for i in range(0, len(values)):
        values[i] = (np.array(values[i]) / float(max_loss)).tolist()

    out_path = '../eval/generation/'
    axes = ['Epoch', 'Normalized Loss']
    graph(values, 'Training Loss', out_path, axes, colors, labels, 'upper right', True)

    # # Collect loss data for discriminator only
    # values = []
    # for i in range(0, len(data)):
    #     for loss_data in data[i]:
    #         losses = []
    #         for loss in loss_data:
    #             losses.append( re.sub('[^0-9]', '', loss) )

    #         if len(values) <= i:
    #             values.append([])
    #         values[i].append(float(losses[3]))

    # for i in range(0, len(values)):
    #     values[i] = values[i][0:(opt.numEpochs)]

    # # Normalize losses
    # max_loss = np.matrix(values).max()
    # for i in range(0, len(values)):
    #     values[i] = (np.array(values[i]) / float(max_loss)).tolist()

    # out_path = '../eval/generation/'
    # axes = ['Epoch', 'Normalized Loss']
    # graph(values, 'Discriminator Training Loss', out_path, axes, colors, labels, 'upper right', True)

    # # Collect time data
    # values = []
    # for i in range(0, len(data)):
    #     times = []
    #     if len(values) <= i:
    #         values.append([])
    #     for time_data in data[int(i)]:
    #         times.append(float(time_data[0]))
    #         time = np.sum( np.array(times, dtype=float) )
    #         values[i].append(time / 3600.0)
    #     print('Gen {:d}: {:0.6f}'.format(2**(6 - i), np.sum(np.array(times, dtype=float)) / float(3600)))

    # for i in range(0, len(values)):
    #     values[i] = values[i][0:(opt.numEpochs)]
    
    # # Graph time data
    # axes = ['Epoch', 'Time (hours)']
    # graph(values, 'Training Time of Generation Models', out_path, axes, colors, labels, 'lower right', False)

def image_from_blocks(write_images):
    table = ''
    eval_root = '../attngan/output/{}/{}/'.format(opt.inputFolder, opt.inputSpecifier)
    if write_images:
        utils.make_dir(eval_root + 'testing/')
        utils.make_dir(eval_root + 'testing/real/')
        utils.make_dir(eval_root + 'testing/fake/')
    for i in range(0, opt.evalRange):
        if opt.evalRange == 1:
            dim = opt.blockDim
        else:
            dim = int(2**(6 - i))
        table += '\\hline\n\\multicolumn{5}{|c|}{\\textit{Blockdim ' + str(dim) + '}} \\\\\n\\hline\n\\textbf{Frame} & \\textbf{MSE} & \\textbf{SSIM} & \\textbf{PSNR} & \\textbf{L\\textsuperscript{2}} \\\\\n\\hline\n'
        if write_images:
            utils.clear_dir(eval_root + 'testing/real/{}'.format(dim))
            utils.clear_dir(eval_root + 'testing/fake/{}'.format(dim))
        x_res = int(floor(int(opt.xRes) / dim) * dim)
        y_res = int(floor(int(opt.yRes) / dim) * dim)
        x_div = int(x_res / dim)
        y_div = int(y_res / dim)

        # Initialize file variables
        for f in range(7, 11):
            f_original = '{}{:03d}.jpg'.format(opt.targetFrame, f)
            eval_folder = eval_root + '{}/eval/{}/'.format(dim, f)
            utils.clear_dir(eval_folder + 'parts/')
            utils.clear_dir(eval_folder + 'complete/')
            print(eval_folder)

            # Iterate over all captions
            #for c in range(0, int(opt.captions)):
            avg = []
            img_original = cv2.resize(cv2.imread(f_original), (x_res, y_res)) # cv2.imread(f_original)[0 : y_res, 0 : x_res]
            #img_out = np.full((y_res, x_res * 2, 3), 0, dtype=int)
            img_out = np.full((y_res, x_res, 3), 0, dtype=int)
            index = 1

            # Iterate over image space
            for row in range(0, y_div):
                data = []
                for col in range(0, x_div):
                    # Find input index and store image
                    index = row * x_div + col + 1
                    if opt.srgan:
                        f_in = eval_folder + '{}/{}.png'.format(opt.inputPrefix, index) # resolution
                    else:
                        f_in = eval_folder + '{}/{}/0_s_0_g1.png'.format(opt.inputPrefix, index) # generation
                    if not f_in or not os.path.exists(f_in):
                        data.append([0, 0, 0])
                        continue

                    # Place image in output
                    img_in = cv2.resize(cv2.imread(f_in), (dim, dim))
                    x_offset = col * dim
                    y_offset = row * dim
                    img_out[y_offset : (y_offset + dim), x_offset : (x_offset + dim)] = img_in
                    data.append(img_in.mean(axis=0).mean(axis=0))
                    utils.evaluate_images(eval_folder + 'parts/', img_original[y_offset : (y_offset + dim), x_offset : (x_offset + dim)], img_in, '{:d}'.format(index), False)
                    if write_images:
                        f_out = eval_root + 'testing/fake/{}/{}_{}.png'.format(dim, f, index)
                        cv2.imwrite(f_out, img_in)
                        f_out = eval_root + 'testing/real/{}/{}_{}.png'.format(dim, f, index)
                        cv2.imwrite(f_out, img_original[y_offset : (y_offset + dim), x_offset : (x_offset + dim)])
                avg.append(data)

            # Write full-resolution images
            #img_out[0 : y_res, x_res : x_res * 2] = img_original
            #img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
            img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res, y_res))
            table_entry = utils.evaluate_images(eval_folder + 'complete/', img_original, img_out, 'complete', write_images)
            print('Wrote image: ' + eval_folder + 'complete/')

            table += '{:d}'.format(f)
            for i in range(0, len(table_entry)):
                table += ' & {:0.3f}'.format(table_entry[i])
            table += ' \\\\\n\\hline\n'

        # Write average-resolution images
        # f_out = eval_folder + '{}_frame_avg.png'.format(c)
        # img_out = np.full((y_div, x_div * 2, 3), 0, dtype=int)
        # img_in = np.array(avg, dtype='uint8')
        # img_out[0 : y_div, 0 : x_div] = img_in
        # img_original = cv2.resize(cv2.imread(f_original), (x_div, y_div))
        # img_out[0 : y_div * dim, x_div : (x_div * 2)] = img_original
        # img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res * 2, y_res))
        # cv2.imwrite(f_out, img_out)
        # print('Wrote image: ' + f_out)

    print(table)

def generate_bird_filenames():
    filenames = ''
    eval_root = '../attngan/data/birds/'
    eval_folder = eval_root + 'text/'

    class_names = os.listdir(eval_folder)
    for class_name in class_names:
        image_names = os.listdir(eval_folder + class_name + '/')
        for image_name in image_names:
            if len(filenames) > 0:
                filenames += '\n'
            filenames += 'text/{}/{}'.format(class_name, image_name.replace('.txt', ''))

    f_out = open(eval_root + 'example_filenames.txt', 'w')
    f_out.write(filenames)
    f_out.close()

def evaluate_birds(write_images):
    dim = opt.blockDim
    eval_root = '../attngan/output/{}/{}/'.format(opt.inputFolder, opt.inputSpecifier)
    if write_images:
        utils.make_dir(eval_root + 'training/')
        utils.make_dir(eval_root + 'training/real/')
        utils.make_dir(eval_root + 'training/fake/')
    if write_images:
        utils.clear_dir(eval_root + 'training/real/')
        utils.clear_dir(eval_root + 'training/fake/')
    x_res = opt.blockDim
    y_res = opt.blockDim
    input_folder = opt.targetFrame
    eval_folder = eval_root + 'eval/'
    utils.clear_dir(eval_folder + 'parts/')
    print(eval_folder)

    class_names = os.listdir(input_folder)
    for class_name in class_names:
        image_names = os.listdir(input_folder + class_name + '/')
        for image_name in image_names:
            # Initialize file variables
            f_original = '{}{}/{}'.format(opt.targetFrame, class_name, image_name)

            # Iterate over all captions
            bird_name = image_name[:-4]
            for c in range(0, int(opt.captions)):
                f_in = eval_folder + 'output/{}/0_s_{}_g1.png'.format(bird_name, c)
                if not f_in or not os.path.exists(f_in):
                    continue

                img_original = cv2.imread(f_original)[0 : y_res, 0 : x_res]
                img_out = cv2.imread(f_in)[0 : y_res, 0 : x_res]

                # Eval and write images
                img_original = cv2.resize(np.array(img_original, dtype='uint8'), (x_res, y_res))
                img_out = cv2.resize(np.array(img_out, dtype='uint8'), (x_res, y_res))
                utils.evaluate_images(eval_folder + 'parts/', img_original, img_out, 'parts', write_images)
                if write_images:
                    f_out = eval_root + 'training/fake/{}_{}.png'.format(dim, bird_name, c)
                    cv2.imwrite(f_out, img_in)
                    f_out = eval_root + 'training/real/{}_{}.png'.format(dim, bird_name, c)
                    cv2.imwrite(f_out, img_original)

def compare_images(img_path_1, img_path_2, img_str_out):
    eval_folder = '../attngan/output/{}/analysis/'.format(opt.inputFolder)
    img_1 = cv2.imread(eval_folder + img_path_1)
    img_2 = cv2.imread(eval_folder + img_path_2)
    print(utils.evaluate_images(eval_folder, img_1, img_2, img_str_out, True))

parser = argparse.ArgumentParser()
parser.add_argument('--inputFolder', default='bunny', help='input folder for evaluation') # birds
parser.add_argument('--inputSpecifier', default='2f_glass', help='input GAN path specifier') # birds, Exp_final
parser.add_argument('--evalRange', type=int, default=1, help='dimension range to iterate over')
parser.add_argument('--numEpochs', type=int, default=250, help='input number of epochs')
parser.add_argument('--numFrames', type=int, default=30, help='input number of frames')
parser.add_argument('--inputPrefix', default='output', help='input GAN path prefix')
parser.add_argument('--targetFrame', default='../datasets/Bunny_Glass/images/', help='target source') # ../attngan/data/birds/dataset/images/
parser.add_argument('--blockDim', type=int, default=4, help='dimension of frameblocks')
parser.add_argument('--captions', type=int, default=10, help='number of captions to output')
parser.add_argument('--xRes', type=int, default=540, help='resolution of target image in the x dimension')
parser.add_argument('--yRes', type=int, default=540, help='resolution of target image in the y dimension')
parser.add_argument('--srgan', type=bool, default=False, help='switch to resolution case')

opt = parser.parse_args()
print(opt)

#evaluate_pretraining()
#evaluate_gentraining()
image_from_blocks(True)

#generate_bird_filenames()
#evaluate_birds(False)

#compare_images('16_all_50.png', '16_all_50.png', 'same')
#compare_images('16_blocks_50.png', '16_real.png', 'blocks')