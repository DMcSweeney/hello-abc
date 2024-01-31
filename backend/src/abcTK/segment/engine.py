"""
Base class for segmentation engine 
"""

import os
import ast
import time
import logging
import SimpleITK as sitk
import numpy as np

import torch
import onnxruntime as ort
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from monai.transforms.spatial.functional import resize
from scipy.special import softmax
import skimage
from flask import abort

from abcTK.segment.writer import sanityWriter

logger = logging.getLogger(__name__)


class segmentationEngine():
    def __init__(self, output_dir, modality, vertebra, worldmatch_correction, fat_threshold=(-190, -30), muscle_threshold=(-29, 150), **kwargs):
        self.output_dir = output_dir
        self.modality = modality
        self.v_level = vertebra
        
        self.worldmatch_correction = worldmatch_correction  ## If data has gone through worldmatch need to shift intensities by -1024

        self.thresholds = {
            'skeletal_muscle': muscle_threshold,
            'subcutaneous_fat': fat_threshold,
            'IMAT': fat_threshold,
            'visceral_fat': fat_threshold,
            'body': (None, None) # No thresholds for body mask
        }

        self._init_model_bank() #* Load bank of models
        self._set_options() #* Set ONNX session options

        self.ort_session = ort.InferenceSession(self.model_paths[modality], sess_options=self.sess_options)
        
        
        #* ImageNet pre-processing transforms
        self.transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225), max_pixel_value=1),
            ToTensorV2()
            ],)
        
    
    def forward(self, input_path, slice_number, num_slices, loader_function, generate_bone_mask, **kwargs):
        ###* ++++++++++ PRE-PROCESS +++++++++++++++++
        mask_dir = os.path.join(self.output_dir, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        start = time.time()

        #* Load input volume
        Image = loader_function(input_path) # Returns SimpleITK image and reference slice
        Image, orient = self.reorient(Image, orientation='LPS')

        if orient.GetFlipAxes()[-1]:
            slice_number = Image.GetSize()[-1] - int(slice_number) - 1 # Since size starts at 1 but indexing starts at 0

        if type(Image) == np.array:
            image = Image
            pixel_spacing = (1, 1, 1)
        else:
            image = sitk.GetArrayFromImage(Image)
            pixel_spacing = Image.GetSpacing()
        self.image = image.copy()
        
        ##* Load the bone mask or generate
        if type(generate_bone_mask) == bool and generate_bone_mask:
            # Regenerat
            logger.info("Generating bone mask")
            Bone = self.generate_bone_mask(Image, pixel_spacing)
            sitk.WriteImage(Bone, os.path.join(mask_dir, 'BONE.nii.gz'))
            Bone, _ = self.reorient(Bone, orientation='LPS')
            self.bone = sitk.GetArrayFromImage(Bone)
        elif type(generate_bone_mask) == str:
            # Assume this is a path
            logger.info(f"Reading bone mask from file: {generate_bone_mask}")
            Bone = sitk.ReadImage(generate_bone_mask)
            Bone, _ = self.reorient(Bone, orientation='LPS')
            self.bone = sitk.GetArrayFromImage(Bone)
        self.bone = np.logical_not(self.bone)

        #* Create some holders to put predictions
        #TODO Should predictions be written to disk? Slower but less mem. w/ big input res.
        holder = np.zeros_like(image, dtype=np.int8)
        self.holders = {
            'skeletal_muscle': holder.copy(),
            'subcutaneous_fat': holder.copy(), 
            'visceral_fat': holder.copy(),
            'IMAT': holder.copy(),
            'body': holder.copy()
        }

        #* Subset the reference image
        #TODO check this works as expected with num_slices = 0
        self.slice_number = slice_number ## Reference slice
        self.num_slices = num_slices
        self.img = self.prepare_multi_slice(image, slice_number, num_slices)

        pre_processing_time = time.time() - start
        logger.info(f"Pre-processing: {pre_processing_time} s")

        ###* ++++++++++ INFERENCE +++++++++++++++++
        #TODO This can be parallelised
        chan2_outputs = []
        chan3_outputs = []
        logger.info(f"==== INFERENCE ====")
        start = time.time()
        for i in range(self.img.shape[0]):
            # Returns True if ROI in channel else False
            chan1, chan2, chan3 = self.per_slice_inference(i)
            chan2_outputs.append(chan2)
            chan3_outputs.append(chan3)

        inference_time = time.time() - start
        logger.info(f"Total inference: {inference_time} s")

        ###* ++++++++++ POST-PROCESS +++++++++++++++++
        start = time.time()
        ## Extracts IMAT
        self.extract_imat(image)
        
        #TODO IF a mask exists, read it, overwrite and save!! That way all levels will be annotated in the same file.
        output = self.post_process(mask_dir, Image, chan2_outputs, chan3_outputs) # Returns stats for every compartment at every slice
        post_processing_time = time.time() - start
        logger.info(f"Post-processing: {post_processing_time} s")
        return output 

    ###############################################
    #* ================ HELPERS ==================
    ###############################################
        

    def generate_bone_mask(self, Image, pixel_spacing, threshold = 350, radius = 3):
        #~ Create bone mask (by thresholding) for handling partial volume effect
        #@threshold in HU; radius in mm.
        logger.info(f"Generating bone mask using threshold ({threshold}) and expanding isotropically by {radius} mm")
        #* Apply threshold
        bin_filt = sitk.BinaryThresholdImageFilter()
        bin_filt.SetOutsideValue(1)
        bin_filt.SetInsideValue(0)

        if self.worldmatch_correction:
            bin_filt.SetLowerThreshold(0)
            bin_filt.SetUpperThreshold(threshold+1024)
        else:
            bin_filt.SetLowerThreshold(-1024)
            bin_filt.SetUpperThreshold(threshold)

        bone_mask = bin_filt.Execute(Image)
        
        #* Convert to pixels
        pix_rad = [int(radius//elem) for elem in pixel_spacing]
        
        #* Dilate mask
        dil = sitk.BinaryDilateImageFilter()
        dil.SetKernelType(sitk.sitkBall)
        dil.SetKernelRadius(pix_rad)
        dil.SetForegroundValue(1)
        Bone= dil.Execute(bone_mask)
        return sitk.Cast(Bone, sitk.sitkInt8)
    
    def prepare_multi_slice(self, image, slice_number, num_slices):
        """
        This prepares multi-slice inputs and maintains slice # to index mapping.
        i.e. Batch size = 2*num_slices + 1  
        """
        self.idx2slice = {}
        logger.info(f"Pre-processing input slices (# slices: {num_slices*2 +1})")
        for i, slice_ in enumerate( np.arange(slice_number- num_slices, slice_number+num_slices+1) ):
            im_tensor = self.pre_process(image, slice_)

            if im_tensor is None:
                logger.warn(f"The selected slice ({slice_}) is out of range - skipping.")
                continue

            if i == 0:
                img = im_tensor[None]
            else:
                img = torch.cat([img, im_tensor[None]], axis=0)
            self.idx2slice[i] = slice_

        return np.array(img)
    
    def pre_process(self, image, slice_number):
        #* Pre-processing
        try:
            im = image[slice_number]
        except IndexError: #* Slice out of bounds
            return None
        
        if self.worldmatch_correction:
            logging.info("Applying worldmatch correction (-1024 HU)")
            im -= 1024

        self.settings = self._get_window_level()
        logging.info(f"Window/Level ({self.settings['window']}/{self.settings['level']}) normalisation")
        im = self.wl_norm(im, window=self.settings['window'], level=self.settings['level'])
        
        logging.info(f"Converting input to three channels")
        im = self.expand(im) #* 3 channels
        logging.info(f"Applying transforms: {self.transforms}")
        augmented = self.transforms(image=im)
        return augmented['image']
    
    def split_predictions(self, pred):
        #* Split prediction into components since different models gen. diff outputs
        if pred.shape[0] == 4: #background,muscle,sub,visc
            return pred[1], pred[2], pred[3]
        elif pred.shape[0] == 3: #background, muscle, body mask
            return pred[1], pred[2], None
        else:
            return pred[1], None, None
    
    def remove_bone(self, i, pred):
        # Resize to match prediction/input
        bone_mask = self.bone[self.idx2slice[i]]
        return np.logical_and(pred, bone_mask)   

    def per_slice_inference(self, i):
        input = self.img[[i]] 
        is_divisible = [True if x % 2 == 0 else False for x in input.shape[-2:] ]
        is_too_small = [True if x < 256 else False for x in input.shape[-2:]]
        if not all(is_divisible) or all(is_too_small):
            #TODO Resampling will affect measurements, if new pixel size not used for calc.
            logger.error(f"Issues with input shape: {input.shape}, resampling not yet implemented.")
            return None, None, None
            #input = resize(input, (512, 512), mode='bicubic')

        prediction = self.inference(input)
        logger.info(f"Splitting prediction (shape: {prediction.shape}) into compartments.")
        chan1, chan2, chan3 = self.split_predictions(prediction) # Split prediction into compartments#
    
        self.holders['skeletal_muscle'][self.idx2slice[i]] = self.remove_bone(i, chan1) if self.bone is not None else chan1

        if chan2 is not None and chan3 is not None:
            logger.info("Fat segmentations detected - adding")
            self.holders['subcutaneous_fat'][self.idx2slice[i]] = self.remove_bone(i, chan2) if self.bone is not None else chan2
            self.holders['visceral_fat'][self.idx2slice[i]] = self.remove_bone(i, chan3) if self.bone is not None else chan3
            return True, True, True # Muscle/SF/VF
        
        elif chan2 is not None and chan3 is None: #Muscle/Body
            logger.info("Body mask detected - adding")
            self.holders['body'][self.idx2slice[i]] = self.remove_bone(i, chan2) if self.bone is not None else chan2
            return True, True, False

        return True, False, False

    def inference(self, img):
        #* Forward pass through the model
        t= time.time()
        ort_inputs = {self.ort_session.get_inputs()[0].name: \
            img.astype(np.float32)}
        logging.info(f'Model load time (s): {np.round(time.time() - t, 7)}')
        #* Inference
        t= time.time()
        outputs = np.array(self.ort_session.run(None, ort_inputs)[0])
        outputs = np.squeeze(outputs)
        logging.info(f'Inference time (s): {np.round(time.time() - t, 7)}')
        logging.info(f"Model outputs: {outputs.shape}")

        if outputs.shape[0] in [3, 4]:
            logging.info("Multiple channels detected, applying softmax")
            pred = np.argmax(softmax(outputs, axis=0), axis=0).astype(np.int8) # Argmax then one-hot encode
            preds = [np.where(pred == val, 1, 0) for val in np.unique(pred)] # one-hot encode
            return np.stack(preds)
        else:
            logging.info("Single channel detected, applying sigmoid")
            return np.round(self.sigmoid(outputs)).astype(np.int8)

    def post_process(self, output_dir, refImage, chan2_outputs, chan3_outputs):
        writer = sanityWriter(self.output_dir, self.slice_number, self.num_slices)

        ## Convert predictions to ITK images and save
        data = {}
        #####==========  IMAT ====================
        logger.info("Extracting IMAT stats")
        data['IMAT'] = self.extract_stats(self.holders['IMAT'], self.thresholds['IMAT'])
        logger.info("Writing IMAT sanity check")
        writer.write_segmentation_sanity('IMAT', self.image, self.holders['IMAT'])
        #* Convert predictions back to ITK Image, using input Image as reference
        logger.info(f"Converting IMAT mask to ITK Image. Size: {self.holders['IMAT'].shape}")
        IMAT  = self.npy2itk(self.holders['IMAT'] , refImage)
        logger.info("Writing IMAT mask")
        self.save_prediction(output_dir, 'IMAT', IMAT)

        ##### =========== MUSCLE =================
        #* Remove IMAT from muscle mask
        logger.info("Extracting skeletal muscle stats")
        skeletal_muscle = np.where(self.holders['IMAT'] == 1, 0, self.holders['skeletal_muscle']).astype(np.int8)
        data['skeletal_muscle'] = self.extract_stats(self.holders['skeletal_muscle'], self.thresholds['skeletal_muscle'])
        logger.info("Writing skeletal muscle sanity check")
        writer.write_segmentation_sanity('MUSCLE', self.image, self.holders['skeletal_muscle'])
        logger.info(f"Converting skeletal muscle mask to ITK Image. Size: {skeletal_muscle.shape}")
        SkeletalMuscle = self.npy2itk(skeletal_muscle, refImage)
        logger.info("Writing skeletal muscle mask")
        self.save_prediction(output_dir,'MUSCLE', SkeletalMuscle)

        ##### =========== SUBCUT/VISCERAL FAT =================
        #* Repeat with subcut and visceral fat. if they exist
        if any(chan2_outputs) and any(chan3_outputs):
            logger.info("Extracting subcutaneous fat stats")
            data['subcutaneous_fat'] = self.extract_stats(self.holders['subcutaneous_fat'], self.thresholds['subcutaneous_fat'])
            logger.info("Writing subcutaneous fat sanity check")
            writer.write_segmentation_sanity('SUBCUT_FAT', self.image, self.holders['subcutaneous_fat'])
            logger.info(f"Converting subcutaneous fat mask to ITK Image. Size: {self.holders['subcutaneous_fat'].shape}")
            SubcutaneousFat = self.npy2itk(self.holders['subcutaneous_fat'], refImage)
            
            logger.info("Extracting visceral fat stats")
            data['visceral_fat'] = self.extract_stats(self.holders['visceral_fat'], self.thresholds['visceral_fat'])
            logger.info("Writing visceral fat sanity check")
            writer.write_segmentation_sanity('VISCERAL_FAT', self.image, self.holders['visceral_fat'])
            logger.info(f"Converting visceral fat mask to ITK Image. Size: {self.holders['visceral_fat'].shape}")
            VisceralFat = self.npy2itk(self.holders['visceral_fat'], refImage)
            
            logger.info("Writing subcutaneous fat mask")
            self.save_prediction(output_dir, 'SUBCUT_FAT', SubcutaneousFat)
            logger.info("Writing visceral fat mask")
            self.save_prediction(output_dir, 'VISCERAL_FAT', VisceralFat)

        ##### =========== BODY MASK =================
        elif any(chan2_outputs) and not any(chan3_outputs):
            logger.info("Extracting body stats")
            data['body'] = self.extract_stats(self.holders['body'], self.thresholds['body'])
            logger.info("Writing body sanity check")
            writer.write_segmentation_sanity('BODY', self.image, self.holders['body'])
            logger.info(f"Converting body mask to ITK Image. Size: {self.holders['body'].shape}")
            Body = self.npy2itk(self.holders['body'], refImage)
            logger.info("Writing body mask")
            self.save_prediction(output_dir, 'BODY', Body)
        else:
            logger.warning(f"No predictions other than skeletal muscle")
        return data

    def extract_stats(self, mask, thresholds):
        #* Extract region of interest
        prediction = mask[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]
        image = self.image[self.slice_number-self.num_slices:self.slice_number+self.num_slices+1]

        #* Apply thresholding
        if not all([x is None for x in thresholds]):
            threshold_image = np.logical_and(
                image >= thresholds[0], ## Threshold the subset image
                image <= thresholds[1],
            ).astype(np.int8)

            prediction = np.logical_and(threshold_image, prediction).astype(np.int8)

        #* Calculate stats across subset
        stats = {}
        slice_numbers = [x for x in np.arange(self.slice_number-self.num_slices, self.slice_number+self.num_slices+1) ]

        for idx, slice_num in zip(range(image.shape[0]), slice_numbers) :
            im, pred = image[idx], prediction[idx]
            #* Area calculation          
            area = float(np.sum(pred))
            #* Density calculation
            density = np.mean(im[pred==1])
            stats[f'Slice {slice_num}'] = {'area (voxels)': area, 'density (HU)': density}
        
        return stats

    def extract_imat(self, numpyImage):
        # Extract IMAT from muscle segmentation: fatByThreshold U muscleSegmentation

        logging.info(f"Generating IMAT mask using thresholds: {self.thresholds['IMAT']}")
        blurred_image = skimage.filters.gaussian(numpyImage, sigma=0.5, preserve_range=True)
        fat_threshold = np.logical_and(
            blurred_image >= self.thresholds['IMAT'][0],
            blurred_image <= self.thresholds['IMAT'][1]
            ).astype(np.int8)
        IMAT = np.logical_and(fat_threshold, self.holders['skeletal_muscle']).astype(np.int8)
        self.holders['IMAT'] = skimage.measure.label(IMAT, connectivity=2, return_num=False) # connected components

    def save_prediction(self, output_dir, tag, Prediction):
        #TODO Convert predictions to RTStruct instead of nii
        #TODO This writes the prediction into the frame of the re-oriented input. If original scan is not LPS, these won't match...
        #* Save mask to outputs folder
        output_filename = os.path.join(output_dir, tag + '.nii.gz')
        logger.info(f"Saving prediction to: {output_filename}")
        sitk.WriteImage(Prediction, output_filename)
    
    ###########################################################
    #* =================== STATIC METHODS =====================
    ############################################################
    @staticmethod
    def wl_norm(img, window, level):
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(img, minval, maxval)
        wld -= minval
        wld /= window
        return wld

    @staticmethod
    def expand(img):
        #* Convert to 3 channels
        return np.repeat(img[..., None], 3, axis=-1)
    
    @staticmethod
    def npy2itk(npy, reference):
        #* npy array to itk image with information from reference
        Image = sitk.GetImageFromArray(npy)
        Image.CopyInformation(reference)
        return Image

    @staticmethod
    def reorient(Image, orientation='LPS'):
        orient = sitk.DICOMOrientImageFilter()
        orient.SetDesiredCoordinateOrientation(orientation)
        return orient.Execute(Image), orient
    
    #####################################################
    #*  ================= OPTIONS ==================
    #####################################################
    def _init_model_bank(self):
        #* Paths to segmentation models
        model_bank = {
            'C3': {'CT': '/src/models/segmentation/c3_pCT.quant.onnx',
                    'CBCT': '/src/models/segmentation/C3_cbct.onnx'},
            'T4': {'CT': '/src/models/segmentation/TitanMixNet-Med-T4-Body-M.onnx'
                    },
            'T9': {'CT': '/src/models/segmentation/TitanMixNet-Med-T9-Body-M.onnx'
                    },
            'T12': {'CT': '/src/models/segmentation/TitanMixNet-Med-T12-Body-M.onnx'
                    },
            'L3': {'CT': '/src/models/segmentation/TitanMixNet-Med-L3-FM.onnx'
            },
            'L5': {'CT': '/src/models/segmentation/TitanMixNet-Med-L5-FM.onnx'
                    },
            'Thigh': {'CT': '/src/models/segmentation/Thigh_14pats.quant.onnx'}
        }
        
        if self.v_level in model_bank:
            self.model_paths = {}
            if self.modality in model_bank[self.v_level]:
                self.model_paths[self.modality] = model_bank[self.v_level][self.modality]
            else:
                raise abort(400, {'message': f'No {self.modality} model for {self.v_level}.'}) 
        else: 
           raise abort(400, {'message': f'Model for {self.v_level} not implemented yet.'})

    def _get_window_level(self):
        #~ Query settings for specific models (window/level)
        settings_bank = {
            'C3': {
                'CT': {'window': 400, 'level': 50}, 
                'CBCT': {'window': 600, 'level': 76}
                },
            'T4': {
                'CT': {'window': 400, 'level': 50}
                },
            'T9': {
                'CT': {'window': 400, 'level': 50}
                },
            'T12': {
                'CT': {'window': 400, 'level': 50}
                },
            'L3': {
                'CT': {'window': 400, 'level': 50}
                },
            'L5': {
                'CT': {'window': 400, 'level': 50}
                },
            'Thigh': {
                'CT': {'window': 400, 'level': 50}
                }
        }

        if self.v_level in settings_bank:
            if self.modality in settings_bank[self.v_level]:
                return settings_bank[self.v_level][self.modality]
            else:
                # Bad request since wrong modality
                raise abort(400, {'message': f'No {self.modality} model for {self.v_level}.'}) 
        else:
            # Bad request since wrong level
            raise abort(400, {'message': f'Model for {self.v_level} not implemented yet.'})

    def _set_options(self):
        #* Inference options
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # ! or ORT_PARALLEL
        self.sess_options.log_severity_level = 4
        self.sess_options.enable_profiling = False
        self.sess_options.inter_op_num_threads = os.cpu_count() - 1
        self.sess_options.intra_op_num_threads = os.cpu_count() - 1