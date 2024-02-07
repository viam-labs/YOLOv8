from typing import ClassVar, Mapping, Sequence, Any, Dict, Optional, Tuple, Final, List, cast
from typing_extensions import Self

from typing import Any, Final, List, Mapping, Optional, Union

from PIL import Image

from viam.media.video import RawImage
from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.resource.types import RESOURCE_NAMESPACE_RDK, RESOURCE_TYPE_SERVICE, Subtype
from viam.utils import ValueTypes


from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision
from viam.components.camera import Camera
from viam.logging import getLogger

from ultralyticsplus import YOLO, postprocess_classify_output
import torch

import time
import asyncio

LOGGER = getLogger(__name__)

class yolov8(Vision, Reconfigurable):
    
    """
    Vision represents a Vision service.
    """
    

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "yolov8")
    
    model: YOLO
    device: str

    # Constructor
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        model = config.attributes.fields["model_location"].string_value
        if model == "":
            raise Exception("A model_location must be defined")
        return

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.DEPS = dependencies
        self.model = YOLO(config.attributes.fields["model_location"].string_value)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
    
        return

    """ Implement the methods the Viam RDK defines for the Vision API (rdk:service:vision) """

    
    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return await self.get_detections(cam_image)
 
    async def get_detections(
        self,
        image: Union[Image.Image, RawImage],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        detections = []
        results = self.model.predict(image, device=self.device)
        if len(results) >= 1:
            index = 0
            for r in results[0].boxes.xyxy:
                detection = { "confidence": results[0].boxes.conf[index].item(), "class_name": results[0].names[results[0].boxes.cls[index].item()], 
                             "x_min": int(r[0].item()), "y_min": int(r[1].item()), "x_max": int(r[2].item()), "y_max": int(r[3].item())}
                detections.append(detection)
                index = index + 1
        return detections
    
    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return await self.get_classifications(cam_image)

    
    async def get_classifications(
        self,
        image: Union[Image.Image, RawImage],
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        results = self.model.predict(image, device=self.device)
        if len(results) >= 1:
            processed_results = postprocess_classify_output(self.model, result=results[0])
            for key in processed_results:
                classifications.append({"class_name": key, "confidence": processed_results[key]})
        return classifications

    
    async def get_object_point_clouds(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        return
    
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        return

