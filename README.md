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

This module needs a few shared libraries that are not bundled inside it:

| Library | Debian / Ubuntu | Fedora / RHEL |
| --- | --- | --- |
| `libGL.so.1` | `libgl1` | `mesa-libGL` |
| `libICE.so.6` | `libice6` | `libICE` |
| `libSM.so.6` | `libsm6` | `libSM` |
| `libX11.so.6` | `libx11-6` | `libX11` |
| `libXext.so.6` | `libxext6` | `libXext` |
| `libglib-2.0.so.0`, `libgthread-2.0.so.0` | `libglib2.0-0t64` (`libglib2.0-0` before Ubuntu 24.04) | `glib2` |
| `libxcb.so.1` | `libxcb1` | `libxcb` |
| `libz.so.1` | `zlib1g` | `zlib-ng-compat` |

These come from the Qt GUI stack that the non-headless `opencv-python` wheel links against, which is why the module needs `libGL` even though it never opens a window. A desktop image usually has them already; a server or minimal container image does not.

`first_run.sh` installs them automatically the first time the module is
installed on a machine, so in most cases there is nothing to do. It supports
`apt`, `dnf`/`yum`, `zypper`, `pacman` and `apk`, picks the right package name
for the distro, and does nothing on macOS.

If it cannot install them — for example the module is not running as root and
passwordless `sudo` is unavailable — it logs the exact command to run by hand
and exits without failing, so it never blocks the rest of the machine from
reconfiguring. Look for `[first_run]` lines in the machine logs. Until the
libraries are present the module fails to start with
`cannot open shared object file`.

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
