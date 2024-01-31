"""
Base class for writing sanity predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

class sanityWriter():
    def __init__(self, output_dir, slice_number, num_slices):
        self.output_dir = os.path.join(output_dir, 'sanity')
        os.makedirs(self.output_dir, exist_ok=True)

        self.slice_number = slice_number
        self.num_slices = num_slices

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

        for i in range(total_slices):
            pred = prediction[i]
            im = self.wl_norm(img[i], 400, 50)

            ax[i].imshow(im, cmap='gray')
            ax[i].imshow(np.where(pred == 0, np.nan, pred), cmap='plasma_r', alpha=0.5)

        output_filename = os.path.join(self.output_dir, tag + '.png')
        fig.savefig(output_filename)


    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld





