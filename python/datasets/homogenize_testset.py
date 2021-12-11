# Removes any attribute/image files which don't have a pair.
# Replace <...> with your file system path(s).
import argparse
import numpy as np
import os as os
import shutil as shutil
import pickle as pickle
import sys as sys
sys.path.append(os.path.abspath('../utils'))
from utils import clear_dir
from utils import make_dir

parser = argparse.ArgumentParser()
parser.add_argument('--guidePath', default='<...>/deep_rendering/python/datasets/<...>/training/', help='input training data path')
parser.add_argument('--inputPath', default='<...>/deep_rendering/python/attngan/output/<...>/testing', help='input SRGAN output data path')
parser.add_argument('--outputPath', default='<...>/deep_rendering/python/attngan/output/<...>/training', help='output testing data path')
parser.add_argument('--startIndex', type=int, default=0, help='starting index of frame files to process')
parser.add_argument('--endIndex', type=int, default=2, help='number of frames to process')
parser.add_argument('--blockDim', type=int, default=4, help='dimension of frameblocks')
parser.add_argument('--blockOffset', type=float, default=1, help='offset for blocks, > 1 blocks will overlap')
parser.add_argument('--prob', type=float, default=0.1, help='probability for missing block inclusion image')

opt = parser.parse_args()
print(opt)

postfix = ''
if float(opt.blockOffset) != 1:
    if int(opt.blockOffset) == float(opt.blockOffset):
        postfix = '_{}'.format(int(opt.blockOffset))
    else:
        postfix = '_{}'.format(float(opt.blockOffset)).replace('.', '-')
blocks_path = opt.guidePath + str(opt.blockDim) + postfix + '/blocks/'
attributes_path = opt.guidePath + str(opt.blockDim) + postfix + '/attributes/'
attrs = os.listdir(attributes_path)
blocks = os.listdir(blocks_path)
attrs.sort()
blocks.sort()
prob = opt.prob

make_dir(opt.outputPath)
make_dir(opt.outputPath + '/real')
make_dir(opt.outputPath + '/fake')
make_dir(opt.outputPath + '/real/' + str(opt.blockDim))
make_dir(opt.outputPath + '/fake/' + str(opt.blockDim))
clear_dir(opt.outputPath + '/real/' + str(opt.blockDim))
clear_dir(opt.outputPath + '/fake/' + str(opt.blockDim))
index = 0
frame = 1
for folder in attrs:
    if index < opt.startIndex:
        continue
    if index >= opt.endIndex:
        break

    attrs_path = attributes_path + folder
    block_path = blocks_path + folder
    attr_files = os.listdir(attrs_path)

    # Check if each attr file exists in blocks folder path
    stored = False
    for f_attrs in attr_files:
        f_block = f_attrs.replace('.txt', '.jpg')
        try:
            block_files = os.listdir(block_path)
        except:
            continue

        # Add the file to the filenames list and copy to output location
        flip = np.random.uniform(0, 1)
        if f_block in block_files or flip <= prob:
            stored = True
            fake_in = '{}/fake/{}/{}_{}.png'.format(opt.inputPath, opt.blockDim, frame, f_attrs.replace('.txt', ''))
            fake_out = '{}/fake/{}/{}_{}.png'.format(opt.outputPath, opt.blockDim, frame, f_attrs.replace('.txt', ''))
            real_in = '{}/real/{}/{}_{}.png'.format(opt.inputPath, opt.blockDim, frame, f_attrs.replace('.txt', ''))
            real_out = '{}/real/{}/{}_{}.png'.format(opt.outputPath, opt.blockDim, frame, f_attrs.replace('.txt', ''))

            # Copy attrs and block files to output destination
            shutil.copyfile(fake_in, fake_out)
            shutil.copyfile(real_in, real_out)

    if stored:
        index += 1
    frame += 1