import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO
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

# Root of viam-server's data-capture tree. The data manager syncs every file it
# finds under here, not only the .capture files it writes itself, so dropping
# plain images and JSON into a subdirectory is enough to get them uploaded.
# VIAM_HOME mirrors how viam-server resolves the same location.
VIAM_DOT_DIR = os.environ.get("VIAM_HOME") or os.path.join(
    os.path.expanduser("~"), ".viam"
)
CAPTURE_DIR = os.path.join(VIAM_DOT_DIR, "capture")

# Frames already in a standard encoding are saved byte-for-byte; anything else
# (RGBA, depth) is re-encoded to PNG so the file opens in ordinary image tools.
MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


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
    confidence: float
    save_detections: bool
    save_dir: str

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
        self.confidence = float(attrs.get("confidence", 0.25))

        self.save_detections = bool(attrs.get("save_detections", False))
        self.save_dir = ""
        if self.save_detections:
            self.save_dir = os.path.expanduser(
                str(attrs.get("save_dir", ""))
                or os.path.join(CAPTURE_DIR, "yolov8", config.name)
            )
            os.makedirs(self.save_dir, exist_ok=True)
            self.logger.info(f"Saving detection frames and sidecars to {self.save_dir}")

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

        if "confidence" in config.attributes.fields:
            confidence = config.attributes.fields["confidence"].number_value
            if not 0 < confidence <= 1:
                raise Exception(
                    f"confidence must be in (0, 1]; got {confidence}"
                )

        if "save_detections" in config.attributes.fields:
            field = config.attributes.fields["save_detections"]
            if field.WhichOneof("kind") != "bool_value":
                raise Exception("save_detections must be a boolean")

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
        image = await self.get_cam_image(camera_name)
        return await self.detect(image, camera_name or self.camera_name)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        return await self.detect(image)

    async def detect(self, image: ViamImage, camera_name: str = "") -> List[Detection]:
        """Run detection on one frame, saving it afterwards when configured to.

        Every entry point that produces detections funnels through here, so
        saving covers the whole detection API rather than a single method.
        """
        detections = self.predict_detections(image)
        if self.save_detections:
            await self.save_capture(image, detections, camera_name)
        return detections

    def predict_detections(self, image: ViamImage) -> List[Detection]:
        detections = []
        results = self.model.predict(
            viam_to_pil_image(image),
            device=self.device,
            classes=self.class_indices,
            conf=self.confidence,
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
        result.detections = await self.detect(
            result.image, camera_name or self.camera_name
        )
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

    async def save_capture(
        self, image: ViamImage, detections: List[Detection], camera_name: str
    ) -> None:
        """Write one frame and its detections into `save_dir`.

        Errors are logged rather than raised: saving is a side effect of the
        detection call and must never be able to fail the call itself.
        """
        try:
            image_bytes, ext = self.encode_image(image)
            now = datetime.now(timezone.utc)
            # Colons are illegal in Windows filenames, so the stem spells the
            # timestamp with dashes; the sidecar carries the real ISO-8601 one.
            # The uuid suffix keeps concurrent calls from colliding within the
            # same microsecond.
            stem = f"{now.strftime('%Y-%m-%dT%H-%M-%S-%f')}_{uuid.uuid4().hex[:8]}"
            image_file = stem + ext
            payload = {
                "captured_at": now.isoformat(),
                "service_name": self.name,
                "camera_name": camera_name or None,
                "image_file": image_file,
                "width": image.width,
                "height": image.height,
                "detections": detections,
            }
            await asyncio.to_thread(
                write_capture_files,
                os.path.join(self.save_dir, image_file),
                image_bytes,
                os.path.join(self.save_dir, stem + ".json"),
                json.dumps(payload, indent=2).encode("utf-8"),
            )
        except Exception as err:
            self.logger.error(
                f"Failed to save detection capture to {self.save_dir}: {err}"
            )

    def encode_image(self, image: ViamImage) -> Tuple[bytes, str]:
        """Return the bytes to save for `image` and the extension to save them under."""
        # A ViamImage mime type can carry a "+lazy" suffix marking deferred
        # encoding; only the base type matters for picking an extension.
        mime = str(image.mime_type).split("+", 1)[0]
        ext = MIME_TO_EXT.get(mime)
        if ext is not None:
            return image.data, ext
        buffer = BytesIO()
        viam_to_pil_image(image).save(buffer, format="PNG")
        return buffer.getvalue(), ".png"

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


def write_capture_files(
    image_path: str, image_bytes: bytes, json_path: str, json_bytes: bytes
) -> None:
    """Write a frame and its sidecar, image first.

    The sidecar goes last so that its presence guarantees the image beside it
    is complete, however the two are picked up for sync.
    """
    write_atomic(image_path, image_bytes)
    write_atomic(json_path, json_bytes)


def write_atomic(path: str, data: bytes) -> None:
    """Write `data` to `path` via a temp file and a rename.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as tmp_file:
        tmp_file.write(data)
    os.replace(tmp_path, path)


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

