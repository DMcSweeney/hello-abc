"""
Base class for writing sanity predictions
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import signal
#
logger= logging.getLogger(__name__)

class sanityWriter():
    def __init__(self, output_dir, slice_number, num_slices):
        self.output_dir = os.path.join(output_dir, 'sanity')
        os.makedirs(self.output_dir, exist_ok=True)

        self.slice_number = slice_number
        self.num_slices = num_slices


    def write_spine_sanity(self, tag, image_path, json, loader_function):
        ## Load image
        Image = loader_function(image_path)
        ## Reorient
        Image, orient = self.reorient(Image, orientation='RPI')
        # Resample
        Image, ratio = self.resample_isotropic_grid(Image)
        image = sitk.GetArrayFromImage(Image)
        image = self.convolve_gaussian(image, axis=-1, sigma=3) ## Remove the ribs!!
        mip = np.max(image, axis=-1)
        ## Plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 7))
        fig.patch.set_facecolor('black')
        #ax.axis('off')
        ax.imshow(mip, cmap='gray')
        ## Scale point and flip if needed
        for vert, coords in json.items():
            logger.info(f"~~~~~~~~~~ COORDS = {coords} ~~~~~~~~~~~~~~~~")
            loc = coords[-1]*ratio[0] ## ratio already switched to npy array indexing
            if orient.GetFlipAxes()[-1]:
                loc = mip.shape[0] - loc - 1 #-1 Since size starts at 1 but indexing at 0

            ax.axhline(loc, c='white', ls='--', linewidth=1.5)
            ax.text(0.95, loc+10, vert, c='white')

        output_filename = os.path.join(self.output_dir, tag + '.png')
        logger.info(f"Writing quality control image to {output_filename}")
        fig.savefig(output_filename)
        return output_filename


    def write_segmentation_sanity(self, tag, image, mask):
        prediction = mask[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]
        img = image[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]

        total_slices = 2*self.num_slices+1
        if total_slices == 1:
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            ax = [ax] # To make subscriptable for plotting
        else:
            if total_slices <= 5:
                fig, ax = plt.subplots(1, total_slices, figsize=(20, 5))
            else:
                fig, axes = plt.subplots(2, total_slices//2, figsize=(20, 10))
                ax = axes.ravel()
        fig.patch.set_facecolor('black')

        slice_nums = np.arange(self.slice_number-self.num_slices, self.slice_number+self.num_slices+1, 1)
        for i in range(total_slices):
            pred = prediction[i]
            im = self.wl_norm(img[i], 400, 50)
            ax[i].set_title(f'Slice: {slice_nums[i]}', c='white', size=20)
            ax[i].imshow(im, cmap='gray')
            ax[i].imshow(np.where(pred == 0, np.nan, pred), cmap='plasma_r', alpha=0.5)

        output_filename = os.path.join(self.output_dir, tag + '.png')
        fig.savefig(output_filename)
        return output_filename

    def write_all_segmentation_sanity(self, tag, image, mask, filter):
        ## Filter is used to figure out what data we expect
        prediction = np.zeros_like(mask['skeletal_muscle'])

        ## Setup figure
        total_slices = 2*self.num_slices+1
        if total_slices == 1:
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            ax = [ax] # To make subscriptable for plotting
        else:
            if total_slices <= 5:
                fig, ax = plt.subplots(1, total_slices, figsize=(20, 5))
            else:
                fig, axes = plt.subplots(2, total_slices//2, figsize=(20, 10))
                ax = axes.ravel()
        fig.patch.set_facecolor('black')

        for i, key in enumerate(filter.keys()):
            ## Merge predictions
            prediction += i*mask[key]

        logger.info(f"PLOTTING {tag} with mask shape: {prediction.shape}")
        img = image[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]
        prediction = prediction[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]

        slice_nums = np.arange(self.slice_number-self.num_slices, self.slice_number+self.num_slices+1, 1)
        for i in range(total_slices):
            pred = prediction[i]
            im = self.wl_norm(img[i], 400, 50)
            ax[i].set_title(f'Slice: {slice_nums[i]}', c='white', size=20)
            ax[i].imshow(im, cmap='gray')
            ax[i].imshow(np.where(pred == 0, np.nan, pred), cmap='plasma_r', alpha=0.5)

        output_filename = os.path.join(self.output_dir, tag + '.png')
        fig.savefig(output_filename)
        return output_filename


    #####################  HELPERS #############
    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld
    
    @staticmethod
    def resample_isotropic_grid(Image, pix_spacing=(1, 1, 1)):
        #* Resampling to isotropic grid
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputDirection(Image.GetDirection())
        resample.SetOutputOrigin(Image.GetOrigin())
        ratio = tuple([x/y for x, y in zip(Image.GetSpacing(), pix_spacing)])
        new_size = [np.round(x*s) \
            for x, s in zip(Image.GetSize(), ratio)]
        resample.SetSize(np.array(new_size, dtype='int').tolist())
        resample.SetOutputSpacing(pix_spacing)
        #* Ratio flipped to account for sitk.Image -> np.array transform
        return resample.Execute(Image), ratio[::-1]

    @staticmethod
    def reorient(Image, orientation='LPS'):
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation)
        return orient.Execute(Image), orient


    @staticmethod
    def convolve_gaussian(image, axis=-1, sigma=3):
        #* Convolve image with 1D gaussian  
        t = np.linspace(-10, 10, 30)
        eps = 1e-24
        gauss = np.exp(-t**2 / (2 * (sigma + eps)**2))
        gauss /= np.trapz(gauss)  # normalize the integral to 1
        kernel = gauss[None, None, :]
        logger.info(f'Kernel size: {kernel.shape}')
        return signal.fftconvolve(image, kernel, mode='same', axes=axis)