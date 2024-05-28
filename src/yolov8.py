from typing import ClassVar, Mapping, Sequence, Any, Dict, Optional, Tuple, Final, List, cast
from typing_extensions import Self

from typing import Any, Final, List, Mapping, Optional, Union

from PIL import Image

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.resource.types import RESOURCE_NAMESPACE_RDK, RESOURCE_TYPE_SERVICE, Subtype
from viam.utils import ValueTypes


from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
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
    
    async def get_cam_image(
        self,
        camera_name: str
    ) -> Image:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return cam_image
    
    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name))
 
    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        detections = []
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
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
        return await self.get_classifications(await self.get_cam_image(camera_name))

    
    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
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

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)
        result.detections = await self.get_detections(result.image)
        result.classifications = await self.get_classifications(result.image, 1)
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False
        )