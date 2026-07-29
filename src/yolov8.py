import os
from pathlib import Path
from typing import ClassVar, Mapping, Any, Optional, List, Sequence, Tuple, cast
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
from viam.utils import struct_to_dict

from ultralytics.engine.results import Results
from ultralytics import YOLO
import torch

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

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        model_location = str(attrs.get("model_location"))

        self.logger.debug(f"Configuring yolov8 model with {model_location}")
        self.DEPS = dependencies
        self.task = str(attrs.get("task")) or None
        self.source_name = attrs.get("source_name") or None
        self.camera_name = str(attrs.get("camera_name", ""))
        self.verbose = bool(attrs.get("verbose", False))

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

        self.class_indices = None
        classes = attrs.get("classes")
        if classes:
            name_to_idx = {name: idx for idx, name in self.model.names.items()}
            unknown = [c for c in classes if c not in name_to_idx]
            if unknown:
                self.logger.error(
                    f"classes {unknown} not found in model; ignoring. "
                    f"Available: {sorted(name_to_idx.keys())}"
                )
            indices = [name_to_idx[c] for c in classes if c in name_to_idx]
            self.class_indices = indices or None

        return self

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        model = config.attributes.fields["model_location"].string_value
        if model == "":
            raise Exception("A model_location must be defined")

        task = config.attributes.fields["task"].string_value
        classes = config.attributes.fields["classes"].list_value
        if classes.values and task not in ("", "detect"):
            raise Exception(
                f"classes is only supported when task is 'detect'; got task='{task}'"
            )

        camera_name = config.attributes.fields["camera_name"].string_value
        required_deps = [camera_name] if camera_name != "" else []

        return required_deps, []

    def get_camera(self, camera_name: str) -> Camera:
        """Resolve a camera the module was given as a dependency.

        The configured `camera_name` is declared as an implicit dependency, so
        viam-server hands it to the module without the machine config naming it
        anywhere else. Cameras listed in the legacy `depends_on` array are also
        present in `DEPS`, so passing one of those by name still resolves.
        """
        name = camera_name or self.camera_name
        if name == "":
            raise Exception(
                "no camera specified: set the camera_name attribute on this service, "
                "or pass a camera name to the *_from_camera method"
            )

        actual_cam = self.DEPS.get(Camera.get_resource_name(name))
        if actual_cam is None:
            raise Exception(
                f"camera '{name}' is not a dependency of this service; "
                f'set "camera_name": "{name}" in the service attributes'
            )
        return cast(Camera, actual_cam)

    async def get_cam_image(self, camera_name: str) -> ViamImage:
        cam = self.get_camera(camera_name)
        if self.source_name:
            cam_images, _ = await cam.get_images(filter_source_names=[self.source_name])
        else:
            cam_images, _ = await cam.get_images()
        return cam_images[0]

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
        results = self.model.predict(
            viam_to_pil_image(image),
            device=self.device,
            classes=self.class_indices,
            verbose=self.verbose,
        )
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
        results = self.model.predict(
            viam_to_pil_image(image), device=self.device, verbose=self.verbose
        )
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
            self.logger.debug(f"Fetching model {self.MODEL_FILE} from {MODEL_URL}")
            urlretrieve(MODEL_URL, self.MODEL_PATH, self.log_progress)

    def log_progress(self, count: int, block_size: int, total_size: int) -> None:
        percent = count * block_size * 100 // total_size
        self.logger.debug(f"\rDownloading {self.MODEL_FILE}: {percent}%")


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

