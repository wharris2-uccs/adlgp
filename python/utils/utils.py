import cv2 as cv2
import glob as glob
#import imutils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os as os
import random
import torch
from skimage.measure import compare_ssim
import sys

def alter_image(img, alpha, beta):
    # Add noise.
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('float32')
    img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0)

    # Gaussian blur.
    img = cv2.GaussianBlur(img, (beta, beta), 0)

    return torch.from_numpy(np.asarray(img).transpose(2, 0, 1))

def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '/*')
        for f in filelist:
            try:
                os.remove(f)
            except:
                if len(os.listdir(f)) > 0:
                    print('rm ' + f)
                    clear_dir(f)
    else:
        os.mkdir(path)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Definition for plotting values.
def plot_together(values, colors, labels, title, axes, path, legend, normalized):
    samples = len(values[0])
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    patches = []
    
    # Plot values and fits.
    for i in range(len(values)):
        avg = np.average(values[i])
        if normalized:
            plt.plot(x_axis, values[i], color=colors[i], alpha=0.2)
            plt.plot(np.unique(x), np.poly1d(np.polyfit(x, values[i], 1))(np.unique(x)), color=colors[i], linestyle='--')
        else:
            plt.axhline(y=values[i][samples - 1], xmin=0, xmax=samples, color=colors[i], linestyle='--')
            plt.plot(x_axis, values[i], color=colors[i], alpha=0.2)
        patches.append(mpatches.Patch(color=colors[i]))
        print(title + '[' + str(i) + ']: ' + str(avg))
    
    # Finish plot.
    plt.legend(patches, labels, loc=legend)
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples - 1])
    if normalized:
        axes.set_ylim([0, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
    
# Definition for plotting loss.
def plot_loss_single(value, color, label, title, path):
    samples = len(value)
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    
    # Plot values.
    avg = np.average(value)
    plt.plot(x_axis, value, color=color, alpha=0.33)
    plt.axhline(y=avg, color=color, xmin=0, xmax=samples, linestyle='--')
    loss_patch = mpatches.Patch(color=color)
    print(title + ': ' + str(avg))
    
    # Finish plot.
    plt.legend([loss_patch], [label], loc='lower right')
    plt.xlabel('Sample')
    plt.ylabel(label)
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    axes.set_ylim([0, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()
    
# Definition for plotting all loss.
def plot_loss_all(values, colors, labels, title, path):
    samples = len(values[0])
    x = [i for i in range(samples)]
    x_axis = np.linspace(0, samples, samples, endpoint=True)
    patches = []
    
    # Plot values.
    for i in range(len(values)):
        plt.plot(x_axis, values[i], color=colors[i], alpha=0.33)
        patches.append(mpatches.Patch(color=colors[i]))
    
    # Plot average lines.
    min = 1
    max = 0
    for i in range(len(values)):
        if min > np.min(values[i]):
            min = np.min(values[i])
        if max < np.max(values[i]):
            max = np.max(values[i])
        avg = np.average(values[i])
        plt.axhline(y=avg, color=colors[i], xmin=0, xmax=samples, linestyle='--', linewidth=0.5)
        
    # Finish plot.
    plt.legend(patches, labels, loc='lower right')
    plt.xlabel('Sample')
    plt.ylabel('Difference Ratio')
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, samples])
    axes.set_ylim([0.725, 1])
    plt.tight_layout()
    
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.clf()

def evaluate_images(eval_folder, img_real, img_fake, img_str_out, write_images):
    # Save output image.
    if write_images:
        cv2.imwrite(eval_folder + img_str_out + '_real.png', img_real)
        cv2.imwrite(eval_folder + img_str_out + '_fake.png', img_fake)

    # img_real = cv2.resize(img_real, (0,0), fx=0.5, fy=0.5) 
    # img_fake = cv2.resize(img_fake, (0,0), fx=0.5, fy=0.5)
    
    # Save evaluation.
    eval_mse = mse(img_real, img_fake)
    eval_ssim = ssim(img_real, img_fake, eval_folder + img_str_out, write_images)
    eval_psnr = psnr(img_real, img_fake)
    eval_simple = simple_distance(img_real, img_fake, eval_folder + img_str_out, write_images)
    eval_distance = distance(img_real, img_fake, eval_folder + img_str_out, write_images)
    f_out = open(eval_folder + 'eval.csv', 'a')
    f_out.write('{}, {}, {}, {}\n'.format( eval_mse, eval_ssim, eval_psnr, eval_distance ))
    f_out.close()
    # print('MSE: {}'.format(mse(img_real, img_fake)))
    # print('PSNR: {}'.format(psnr(img_real, img_fake)))
    # print('SSIM: {}'.format(ssim(img_real, img_fake, img_str_out)))
    # print('Simple L-squared Distance: {}'.format(simple_distance(img_real, img_fake, eval_folder + img_str_out)))
    # print('L-squared Distance: {}'.format(distance(img_real, img_fake, eval_folder + img_str_out)))

    return [eval_mse, eval_ssim, eval_psnr, eval_distance]

# Evaluate two images using MSE.
# Source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/.
def mse(img_real, img_fake):
    return np.mean((img_real - img_fake) ** 2)

# Evaluate two images using SSIM.
# Source: https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python.
# Reference: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-.
def ssim(img_real, img_fake, img_str_out, write_images):
    try:
        # Compute the Structural Similarity Index (SSIM).
        (score, img_out) = compare_ssim(img_real, img_fake, win_size=3, full=True, multichannel=True)
        img_out = (img_out * 255).astype("uint8")

        # Print evaluation and save output image.
        if write_images:
            cv2.imwrite(img_str_out + '_ssim.png', img_out)
        return score
    except ValueError:
        print(ValueError)
        exit()
    except:
        print(sys.exc_info()[0])
        exit()

# Evaluate two images using PSNR.
# Source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/.
def psnr(img_real, img_fake):
    if(mse(img_real, img_fake) == 0):
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse(img_real, img_fake)))

# Evaluate two images using simple bitwise-XOR L-squared.
def simple_distance(img_real, img_fake, img_str_out, write_images):
    # Find minimum height and width.
    height_1, width_1 = img_real.shape[:2]
    height_2, width_2 = img_fake.shape[:2]
    height = min(height_1, height_2)
    width = min(width_1, width_2)

    # Evaluate images.
    img_xor = cv2.bitwise_xor(img_real, img_fake)
    pixel_sum = np.sum(img_xor)
    img_out = cv2.bitwise_not(cv2.cvtColor(img_xor, cv2.COLOR_BGR2GRAY))

    # Print evaluation and save output image.
    if write_images:
        cv2.imwrite(img_str_out + '_simple_L2.png', img_out)
    return 1 - (pixel_sum / (3 * 255.0 * width * height))

# Evaluate two images using L-squared.
def distance(img_real, img_fake, img_str_out, write_images):
    # Find minimum height and width.
    height_1, width_1 = img_real.shape[:2]
    height_2, width_2 = img_fake.shape[:2]
    height = min(height_1, height_2)
    width = min(width_1, width_2)

    # Calculate the pixel distance (L-squared) for each pixel.
    total = 0
    img_out = np.zeros((height, width, 3), np.uint8)
    for h in range(height):
        for w in range(width):
            for i in range(3):
                d = int(np.power(img_real[h][w][i] - img_fake[h][w][i], 2))
                total += d
                img_out[h][w][i] = min(255, d)
    
    # Print evaluation and save output image.
    if write_images:
        cv2.imwrite(img_str_out + '_L2.png', img_out)
    return 1 - (total / (3 * 255.0 * 255.0 * width * height))