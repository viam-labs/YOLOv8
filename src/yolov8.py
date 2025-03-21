import os
from pathlib import Path
from typing import ClassVar, Mapping, Any, Optional, List, cast
from typing_extensions import Self
from urllib.request import urlretrieve

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes


from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.logging import getLogger
from viam.utils import struct_to_dict

from ultralytics.engine.results import Results
from ultralytics import YOLO
import torch

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
)


class yolov8(Vision, EasyResource):
    """
    Vision represents a Vision service.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "yolov8")

    MODEL_FILE = ""
    MODEL_REPO = ""
    MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_REPO))

    model: YOLO
    device: str

    # Constructor
    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        return super().new(config, dependencies)

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ComponentConfig):
        LOGGER.debug("Validating yolov8 service config")
        model = config.attributes.fields["model_location"].string_value
        if model == "":
            raise Exception("A model_location must be defined")
        return []

    # Handles attribute reconfiguration
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)
        model_location = str(attrs.get("model_location"))

        LOGGER.debug(f"Configuring yolov8 model with {model_location}")
        self.DEPS = dependencies
        self.task = str(attrs.get("task")) or None

        if "/" in model_location:
            if self.is_path(model_location):
                self.MODEL_PATH = model_location
            else:
                model_name = str(attrs.get("model_name", ""))
                if model_name == "":
                    raise Exception(
                        "model_name attribute is required for downloading models from HuggingFace."
                    )
                self.MODEL_REPO = model_location
                self.MODEL_FILE = model_name
                self.MODEL_PATH = os.path.abspath(
                    os.path.join(
                        MODEL_DIR,
                        f"{self.MODEL_REPO.replace('/', '_')}_{self.MODEL_FILE}",
                    )
                )

                self.get_model()

            self.model = YOLO(self.MODEL_PATH, task=self.task)
        else:
            self.model = YOLO(model_location, task=self.task)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        return

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        cam_image = await cam.get_image(mime_type="image/jpeg")
        return cam_image

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
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
            result = results[0]
            if result.boxes:
                for r in result.boxes.xyxy:
                    detection = {
                        "confidence": result.boxes.conf[index].item(),
                        "class_name": result.names[result.boxes.cls[index].item()],
                        "x_min": int(r[0].item()),
                        "y_min": int(r[1].item()),
                        "x_max": int(r[2].item()),
                        "y_max": int(r[3].item()),
                    }
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
        count: int = 0,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        results = self.model.predict(viam_to_pil_image(image), device=self.device)
        if len(results) >= 1:
            processed_results = postprocess_classify_output(
                self.model, result=results[0]
            )
            for key in processed_results:
                classifications.append({
                    "class_name": key,
                    "confidence": processed_results[key],
                })
        return classifications

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        pass

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None
    ) -> Mapping[str, ValueTypes]:
        pass

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
            object_point_clouds_supported=False,
        )

    def is_path(self, path: str) -> bool:
        try:
            Path(path)
            return os.path.exists(path)
        except ValueError:
            return False

    def get_model(self):
        if not os.path.exists(self.MODEL_PATH):
            MODEL_URL = f"https://huggingface.co/{self.MODEL_REPO}/resolve/main/{self.MODEL_FILE}"
            LOGGER.debug(f"Fetching model {self.MODEL_FILE} from {MODEL_URL}")
            urlretrieve(MODEL_URL, self.MODEL_PATH, self.log_progress)

    def log_progress(self, count: int, block_size: int, total_size: int) -> None:
        percent = count * block_size * 100 // total_size
        LOGGER.debug(f"\rDownloading {self.MODEL_FILE}: {percent}%")


# vendored and updated from ultralyticsplus library
def postprocess_classify_output(model: YOLO, result: Results) -> dict:
    """
    Postprocesses the output of classification models

    Args:
        model (YOLO): YOLO model
        prob (np.ndarray): output of the model

    Returns:
        dict: dictionary of outputs with labels
    """
    output = {}
    if isinstance(model.names, list):
        names = model.names
    elif isinstance(model.names, dict):
        names = model.names.values()
    else:
        raise ValueError("Model names must be either a list or a dict")

    if result.probs:
        for i, label in enumerate(names):
            output[label] = result.probs[i].item()
        return output
    else:
        return {}

