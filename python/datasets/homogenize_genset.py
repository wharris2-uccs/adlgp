# Generates a dataset from which to train SRGAN 
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
parser.add_argument('--guidePath', default='<...>/deep_rendering/python/datasets/<...>/training', help='input training data path')
parser.add_argument('--inputPath', default='<...>/deep_rendering/python/datasets/<...>/training', help='input SRGAN output data path')
parser.add_argument('--outputPath', default='<...>/deep_rendering/python/attngan/output/<...>/training', help='output testing data path')
parser.add_argument('--startIndex', type=int, default=0, help='starting index of frame files to process')
parser.add_argument('--endIndex', type=int, default=10, help='number of frames to process')
parser.add_argument('--blockDim', type=int, default=64, help='dimension of frameblocks')
parser.add_argument('--blockOffset', type=float, default=1, help='offset for blocks, > 1 blocks will overlap')

opt = parser.parse_args()
print(opt)

postfix = ''
if float(opt.blockOffset) != 1:
    if int(opt.blockOffset) == float(opt.blockOffset):
        postfix = '_{}'.format(int(opt.blockOffset))
    else:
        postfix = '_{}'.format(float(opt.blockOffset)).replace('.', '-')
blocks_path = opt.guidePath + '/' + str(opt.blockDim) + postfix + '/blocks/'
blocks = os.listdir(blocks_path)
blocks.sort()

make_dir(opt.outputPath)
make_dir('{}/{}/'.format(opt.outputPath, opt.blockDim))
make_dir('{}/{}/fake/'.format(opt.outputPath, opt.blockDim))
make_dir('{}/{}/real/'.format(opt.outputPath, opt.blockDim))

frame = 1
for folder in blocks:
    if frame > opt.endIndex:
        break
    block_path = blocks_path + folder
    block_files = os.listdir(block_path)

    # Check if each attr file exists in blocks folder path
    stored = False
    for f_block in block_files:
        # Add the file to the filenames list and copy to output location
        fake_in = '{}/{}/blocks/{:03d}/{}.jpg'.format(opt.inputPath, opt.blockDim, frame, f_block.replace('.jpg', ''))
        fake_out = '{}/{}/fake/{}/{}.png'.format(opt.outputPath, opt.blockDim, frame, f_block.replace('.jpg', ''))
        real_in = '{}/{}/blocks/{:03d}/{}.jpg'.format(opt.guidePath, opt.blockDim, frame, f_block.replace('.jpg', ''))
        real_out = '{}/{}/real/{}/{}.png'.format(opt.outputPath, opt.blockDim, frame, f_block.replace('.jpg', ''))
        make_dir('{}/{}/fake/{}'.format(opt.outputPath, opt.blockDim, frame))
        make_dir('{}/{}/real/{}'.format(opt.outputPath, opt.blockDim, frame))

        # Copy block files to output destination
        shutil.copyfile(fake_in, fake_out)
        shutil.copyfile(real_in, real_out)

    frame += 1