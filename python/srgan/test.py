# Taken from: https://github.com/aitorzip/PyTorch-SRGAN
# python test.py --blockDim 32 --alpha 0.9 --beta 3 --cuda

import argparse
import os as os
import numpy as np
import sys as sys
import utils as utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from keras.datasets import cifar10

from models import Generator, Discriminator, FeatureExtractor

def normalize_images(images):
    return (np.array(images) - np.array(images).min(0)) / np.array(images).ptp(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blockDim', type=int, default=400, help='size of block to use')
    parser.add_argument('--useRange', action='store_true', help='whether or not to use a random range')
    parser.add_argument('--alpha', type=float, default=0.75, help='noise constant to use')
    parser.add_argument('--beta', type=int, default=7, help='blur constant to use')
    parser.add_argument('--alphaPair', type=float, default=0.9, help='noise constant range to use')
    parser.add_argument('--betaPair', type=int, default=3, help='blur constant range to use')
    parser.add_argument('--generation', type=int, default=100, help='epochs to wait between writing images')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--upSampling', type=int, default=1, help='low to high resolution scaling factor')
    parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--generatorWeights', type=str, default='generator_final.pth', help="path to generator weights (to continue training)")
    parser.add_argument('--discriminatorWeights', type=str, default='discriminator_final.pth', help="path to discriminator weights (to continue training)")
    parser.add_argument('--inType', default='bunny', help='input type, one of the following: [frame, cifar]')

    opt = parser.parse_args()
    print(opt)
    
    # pair = None
    # tag = '%d-%d-%d' % (opt.blockDim, int(opt.alpha * 100), opt.beta)
    # if opt.useRange:
    #     opt.alpha = 0.75
    #     opt.beta = 7
    #     pair = [opt.alphaPair, opt.betaPair]
    #     tag = 'range'
    # outf_path = ('%s_outputx%s/' % (opt.inType, str(opt.blockDim)))
    
    inc_path = ('%s_checkpointsx%s/' % (opt.inType, str(opt.blockDim)))
    if not os.path.exists(inc_path):
        print('Error: input checkpoint path %s does not exist. First generate the files using train.py.' % inc_path)
        exit()

    outc_path = ('%s_checkpointsx%s/' % (opt.inType, str(opt.blockDim)))
    outf_path = ('%s_outputx%s/' % (opt.inType, str(opt.blockDim)))
    utils.clear_dir('%shigh_res_real/' % (outf_path))
    utils.clear_dir('%shigh_res_fake/' % (outf_path))
    utils.clear_dir('%slow_res/' % (outf_path))
    dim = opt.blockDim

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # transform = transforms.Compose([transforms.RandomCrop(opt.blockDim),
    #                                 transforms.ToTensor()])

    # normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
    #                                 std = [0.229, 0.224, 0.225])

    transform_x = transforms.Compose([transforms.Resize((int(opt.blockDim/opt.upSampling), int(opt.blockDim/opt.upSampling))), transforms.ToTensor()])
    transform_y = transforms.Compose([transforms.Resize((int((opt.blockDim/opt.upSampling) * opt.upSampling), int((opt.blockDim/opt.upSampling) * opt.upSampling))), transforms.ToTensor()])
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    # Equivalent to un-normalizing ImageNet (for correct visualization)
    #unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
    # unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

    # Load data.
    if opt.inType == 'frame' or opt.inType == 'bunny':
        # transform = transforms.Compose([transforms.RandomCrop(opt.blockDim), transforms.ToTensor()])
        # normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        # data_prefix = '/home/pixarninja/Git/dynamic_frame_generator/python/training/' + str(opt.blockDim) + '/'
        # dataset = datasets.ImageFolder(root=data_prefix + 'testset/', transform=transform)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

        # # Generate testing data.
        # x_test = []
        # y_test = []
        # for i, data in enumerate(dataloader, 0):
        #     high_res_real = data[0]
        #     if np.shape(high_res_real)[0] != opt.batchSize:
        #         continue

        #     # Downsample images to low resolution.
        #     for j in range(opt.batchSize):
        #         x_test.append(utils.alter_image(high_res_real[j].numpy().transpose(1, 2, 0), opt.alpha, opt.beta, pair=pair))
        #         y_test.append(normalize(high_res_real[j]))

        # x_test = torch.stack(x_test)
        # y_test = torch.stack(y_test)

        x_test = []
        y_test = []

        eval_folder = '../attngan/output/bunny/2f_stone/'
        dataset = utils.CustomDataSet(eval_folder + 'training/{}/fake/1'.format(dim), transform_x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False)

        print('Generating fake testing data...')
        for i, data in enumerate(dataloader, 0):
            if i % opt.batchSize == 0:
                print('X: {}/{}'.format(i + 1, len(dataloader)))
            # high_res_fake = data[0]
            # if np.shape(high_res_fake)[0] != opt.batchSize:
            #     continue

            # Downsample images to low resolution.
            for j in range(opt.batchSize):
                if j < len(data):
                    x_test.append(normalize(data[j]))

        dataset = utils.CustomDataSet(eval_folder + 'training/{}/real/1'.format(dim), transform_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False)

        print('Generating real testing data...')
        for i, data in enumerate(dataloader, 0):
            if i % opt.batchSize == 0:
                print('Y: {}/{}'.format(i + 1, len(dataloader)))
            # high_res_real = data[0]
            # if np.shape(high_res_real)[0] != opt.batchSize:
            #     continue

            # Downsample images to low resolution.
            for j in range(opt.batchSize):
                if j < len(data):
                    y_test.append(normalize(data[j]))

        x_test = torch.stack(x_test)
        y_test = torch.stack(y_test)
        print(len(x_test), len(y_test))

    elif opt.inType == 'cifar':
        _, (x_data, _) = cifar10.load_data()

        # Generate testing data.
        x_test = []
        y_test = []
        for i, data in enumerate(x_data, 0):
            data = data / 255.

            # Downsample image to low resolution.
            x_test.append(utils.alter_image(np.array(data), opt.alpha, opt.beta, pair=pair))
            y_test.append(torch.from_numpy(data.transpose(2, 0, 1)))

        x_test = torch.stack(x_test)
        y_test = torch.stack(y_test)

    else:
        print('ERROR: Input data type not recognized')
        exit(1)

    n_samples = int(x_test.shape[0] / opt.batchSize)
    print('\nBatch Size: {}, Batches: {}'.format(opt.batchSize, n_samples))

    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(inc_path + opt.generatorWeights))
    print(generator)

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(inc_path + opt.discriminatorWeights))
    print(discriminator)

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    print(feature_extractor)
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    target_real = Variable(torch.ones(opt.batchSize,1))
    target_fake = Variable(torch.zeros(opt.batchSize,1))

    # if gpu is to be used
    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()

    print('Test started...')
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    # Set evaluation mode (not training)
    generator.eval()
    discriminator.eval()

    for i in range(n_samples):
        low_res = x_test[i * opt.batchSize:(i + 1) * opt.batchSize]
        high_res_real = y_test[i * opt.batchSize:(i + 1) * opt.batchSize]

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))

        high_res_real = high_res_real.float()
        high_res_fake = high_res_fake.float()
        
        ######### Test discriminator #########

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data

        ######### Test generator #########

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), target_real)
        mean_generator_adversarial_loss += generator_adversarial_loss.data

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data

        ######### Status and display #########
        sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i, n_samples,
        discriminator_loss.data, generator_content_loss.data, generator_adversarial_loss.data, generator_total_loss.data))

        for j in range(opt.batchSize):
            if opt.inType == 'frame':
                vutils.save_image(unnormalize(high_res_fake[j]),
                        '%shigh_res_fake/%d.png' % (outf_path, i*opt.batchSize + j + 1),
                        normalize=False)
            else:
                vutils.save_image(high_res_fake[j],
                        '%shigh_res_fake/%d.png' % (outf_path, i*opt.batchSize + j + 1),
                        normalize=True)

            vutils.save_image(high_res_real[j],
                    '%shigh_res_real/%d.png' % (outf_path, i*opt.batchSize + j + 1),
                    normalize=True)
            vutils.save_image(low_res[j],
                    '%slow_res/%d.png' % (outf_path, i*opt.batchSize + j + 1),
                    normalize=True)

    sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i, n_samples,
    mean_discriminator_loss/n_samples, mean_generator_content_loss/n_samples, 
    mean_generator_adversarial_loss/n_samples, mean_generator_total_loss/n_samples))
