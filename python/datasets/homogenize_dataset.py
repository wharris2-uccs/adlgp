# Creates the dataset from which AttnGAN is trained on.
# Replace <...> with your file system path(s).
import argparse
import numpy as np
import os as os
import shutil as shutil
import pickle as pickle
import sys as sys
sys.path.append(os.path.abspath('../utils'))
from utils import clear_dir

parser = argparse.ArgumentParser()
parser.add_argument('--inputPath', default='<...>/deep_rendering/python/datasets/<...>/training/', help='input training data path')
parser.add_argument('--outputPath', default='<...>/deep_rendering/python/attngan/data/<...>/', help='output training data path')
parser.add_argument('--startIndex', type=int, default=0, help='starting index of frame files to process')
parser.add_argument('--endIndex', type=int, default=5, help='number of frames to process')
parser.add_argument('--blockDim', type=int, default=4, help='dimension of frameblocks')
parser.add_argument('--blockOffset', type=float, default=1, help='offset for blocks, > 1 blocks will overlap')
parser.add_argument('--prob', type=float, default=0.8, help='probability for training/testing image')
parser.add_argument('--copyFiles', type=bool, default=True, help='copy files from source locations')

opt = parser.parse_args()
print(opt)

postfix = ''
if float(opt.blockOffset) != 1:
    if int(opt.blockOffset) == float(opt.blockOffset):
        postfix = '_{}'.format(int(opt.blockOffset))
    else:
        postfix = '_{}'.format(float(opt.blockOffset)).replace('.', '-')
blocks_path = opt.inputPath + str(opt.blockDim) + postfix + '/blocks/'
attributes_path = opt.inputPath + str(opt.blockDim) + postfix + '/attributes/'
attrs = os.listdir(attributes_path)
blocks = os.listdir(blocks_path)
attrs.sort()
blocks.sort()
training_list = []
testing_list = []
prob = opt.prob

if opt.copyFiles:
    clear_dir(opt.outputPath + 'text/')
    clear_dir(opt.outputPath + 'images/')
index = 0
for folder in attrs:
    if index < opt.startIndex:
        continue
    if index >= opt.endIndex:
        break

    attrs_path = attributes_path + folder
    block_path = blocks_path + folder
    attr_files = os.listdir(attrs_path)
    if opt.copyFiles:
        clear_dir(opt.outputPath + 'text/' + folder)
        clear_dir(opt.outputPath + 'images/' + folder)

    # Check if each attr file exists in blocks folder path
    stored = False
    for f_attrs in attr_files:
        f_block = f_attrs.replace('.txt', '.jpg')
        try:
          block_files = os.listdir(block_path)
        except:
          continue

        # Add the file to the filenames list and copy to output location
        if f_block in block_files:
            stored = True
            f_item = folder + '/' + f_attrs.replace('.txt', '')
            flip = np.random.uniform(0, 1)
            if flip <= prob:
                training_list.append(f_item)

                if prob == 1.0:
                    testing_list.append(f_item)
            else:
                training_list.append(f_item)
                testing_list.append(f_item)

            # Copy attrs and block files to output destination
            if opt.copyFiles:
                shutil.copyfile(attrs_path + '/' + f_attrs, opt.outputPath + 'text/' + folder + '/' + f_attrs)
                shutil.copyfile(block_path + '/' + f_block, opt.outputPath + 'images/' + folder + '/' + f_block)

    if stored:
        index += 1

# Output the filenames list as a pickle
clear_dir(opt.outputPath + 'train/')
clear_dir(opt.outputPath + 'test/')
with open(opt.outputPath + 'train/filenames.pickle', 'wb') as pfile:
    pickle.dump(training_list, pfile, protocol=0)
with open(opt.outputPath + 'test/filenames.pickle', 'wb') as pfile:
    pickle.dump(testing_list, pfile, protocol=0)