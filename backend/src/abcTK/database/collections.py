"""
Contains schema for the different collections in the database. 
Easier to centralise all of this and push updates during processing
"""

from dataclasses import dataclass

@dataclass
class Images():
    """Class for keeping track of images that have been processed"""
    _id: str
    project: str
    input_path: str
    patient_id: str
    study_uuid: str
    series_uuid: str
    acquisition_date: str
    
    X_spacing: float
    Y_spacing: float
    slice_thickness: float

    labelling_done: bool = False
    segmentation_done: bool = False

@dataclass
class QualityControl():
    """Class for tracking quality control results"""

    _id: str
    project: str
    input_path: str
    patient_id: str
    series_uuid: str

    paths_to_sanity_images: dict
    quality_control: dict
    qc_report: dict 

@dataclass
class Spine():
    """Class for recording outputs of labelling"""
    _id: str
    project: str
    input_path: str
    patient_id: str
    series_uuid: str

    output_dir: str
    prediction: dict
    all_parameters: dict


@dataclass
class Segmentation():
    """Class for monitoring segmentation results"""
    _id: str
    project: str
    input_path: str
    patient_id: str
    series_uuid: str

    output_dir: str
    statistics: dict
    all_parameters: dict