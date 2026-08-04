# YOLOv8 modular service

This module implements the [RDK vision API](https://github.com/rdk/vision-api) in a viam-labs:vision:yolov8 model.

This model leverages the [Ultralytics inference library](https://docs.ultralytics.com/) to allow for object detection and classification from YOLOv8 models.

Both locally deployed YOLOv8 models and models from web sources like [HuggingFace](https://huggingface.co/models?other=yolov8) can be used (HuggingFace models will be downloaded and used locally).

![Example screen recording of usage](./demo.gif)

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `vision` type, then select the `viam-labs:vision:yolov8` model.
Enter a name for your vision and click **Create**.

## System dependencies

This module uses the **headless** OpenCV wheel (`opencv-python-headless`), which
is the same `cv2` API and version as `opencv-python` but is not linked against
Qt/X11. That matters because the GUI build pulls in `libGL.so.1`, `libX11.so.6`,
`libxcb.so.1`, `libICE.so.6`, `libSM.so.6`, `libXext.so.6`, `libglib-2.0.so.0`
and `libgthread-2.0.so.0` at import time — none of which ship on a server or
minimal container image. Without them the module fails to start with
`cannot open shared object file`, which then surfaces as a confusing
`model not registered` cascade on every resource that depends on it.

With the headless wheel the only external library `cv2` needs is `libz.so.1`,
which is part of every Linux base system, so **no extra packages have to be
installed on the machine**.

Keep it headless. If you ever need a GUI-only OpenCV call (`cv2.imshow`,
`cv2.waitKey`, `cv2.namedWindow` and friends), those are unavailable in this
wheel by design — a module has no display to draw on.

`ultralytics` declares a hard dependency on `opencv-python`, so `setup.sh`
replaces it with the headless wheel after installing requirements. That step is
what keeps the shipped bundle free of the Qt linkage.

## Configure your vision service

Copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "model_location": "<string>"
}
```

### Attributes

The following attributes are available for `viam-labs:vision:yolov8` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_location` | string | **Required** |  YOLO model name (such as "yolov8n.pt"), local path to model, or HuggingFace model repo identifier |
| `model_name` | string | Optional |  Name of model file when using HuggingFace repo identifier as `model_location` |
| `camera_name` | string | Optional |  Name of the camera to read from in the `*_from_camera` methods. Declared as an implicit dependency, so the camera is started before this service. |
| `task` | string | Optional |  Name of computer vision task performed by the model: "detect" (default) or "classify" |
| `classes` | list of strings | Optional |  Restrict detections to the listed class names (e.g. `["cup"]`). Names must match `model.names`; unknown names are logged and skipped. Applies only when `task` is `"detect"`. |
| `source_name` | string | Optional |  Image source name to select on multi-source cameras (e.g. `"color"` on an RGBD camera). When omitted, the first image returned by the camera is used. |
| `verbose` | bool | Optional |  Enable Ultralytics' per-prediction logging to stdout. Defaults to `false`. Set to `true` only for debugging — it is very chatty. |

### Example Configurations

YOLO base model:

```json
{
  "model_location": "yolov8n.pt",
}
```

[HuggingFace model](https://huggingface.co/keremberke/yolov8n-hard-hat-detection):

```json
{
  "model_location": "keremberke/yolov8n-hard-hat-detection",
  "model_name": "best.pt"
}
```

Local YOLOv8 model:

```json
{
  "model_location": "/path/to/yolov8n.pt"
}
```

***Note:*** if using the `get_detections_from_camera`, `get_classifications_from_camera` or `capture_all_from_camera` API, set `camera_name` to the camera you want to read from:

```json
{
  "model_location": "yolov8n.pt",
  "camera_name": "cam"
}
```

`viam-server` resolves `camera_name` as an implicit dependency and starts that camera before this service, so there is no need to add a `depends_on` entry.

The `*_from_camera` methods take a camera name as an argument. Leave it empty to use the configured `camera_name`, or pass the name of any camera the service depends on. Existing configurations that list their cameras in `depends_on` continue to work unchanged.

## API

The YOLOv8 resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

### get_classifications(image=*binary*)

### get_classifications_from_camera(camera_name=*string*)
