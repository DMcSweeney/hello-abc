import logging
import os
from typing import Any, Dict, Optional, Union

from monai.networks.nets import SegResNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

from abcTK.spine.engines.vertebra_finder import VertebraFinder

logger = logging.getLogger(__name__)


class find_vertebrae(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "C1": 1,
            "C2": 2,
            "C3": 3,
            "C4": 4,
            "C5": 5,
            "C6": 6,
            "C7": 7,
            "T1": 8,
            "T2": 9,
            "T3": 10,
            "T4": 11,
            "T5": 12,
            "T6": 13,
            "T7": 14,
            "T8": 15,
            "T9": 16,
            "T10": 17,
            "T11": 18,
            "T12": 19,
            "L1": 20,
            "L2": 21,
            "L3": 22,
            "L4": 23,
            "L5": 24,
        }

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            self.path = "/src/models/spine/radiology_segmentation_segresnet_localization_vertebra.pt"
            
        #!! Check if model is actually loaded
        if not os.path.isfile(self.path):
            logger.error("No model file detected!! - aborting")
            raise FileNotFoundError(f"No model file detected - check that the following exists: {self.path}")
        logger.info(f"--------------------- PATH-TO-MODEL ----------------------------- {os.path.abspath(self.path)}")
        self.target_spacing = (1.3, 1.3, 1.3)  # target space for image
        #self.target_spacing = (1.6, 1.6, 1.6)
        #self.target_spacing = (2, 2, 2) 
        # Setting ROI size - This is for the image padding
        self.roi_size = (96, 96, 96)
        #self.roi_size = (64, 64, 64)

        # Network
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(self.labels) + 1,  # labels plus background,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = VertebraFinder(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        pass