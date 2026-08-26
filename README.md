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
passwordless `sudo` is unavailable — it **fails loudly**: it logs the exact
command to run by hand and exits non-zero. That aborts the machine's
reconfiguration, so the machine keeps running its previous, working config
instead of coming up with a module that cannot start. Already-running modules
are left alone. Look for `[first_run]` lines in the machine logs.

Once the libraries are installed the machine picks them up on its next
reconfiguration — no success marker is written on failure, so `first_run` is
retried automatically. Because an aborted reconfiguration is retried every few
seconds, the install attempt itself is rate-limited to once every 10 minutes to
avoid fighting the package-manager lock; the diagnostic and the non-zero exit
still happen on every attempt.

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
| `confidence` | float | Optional |  Minimum detection confidence in (0, 1]; detections below it are discarded. Defaults to `0.25` (the Ultralytics default). |
| `save_detections` | bool | Optional |  Save every frame the detection API sees, along with the detections it produced. Defaults to `false`. See [Saving detections](#saving-detections). |
| `save_dir` | string | Optional |  Where to write saved frames. Defaults to `~/.viam/capture/yolov8/<service name>`, which the data management service syncs to the cloud. Only read when `save_detections` is `true`. |

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

## Saving detections

Set `save_detections` to `true` and the module writes every frame the detection
API sees, together with the detections it produced:

```json
{
  "model_location": "yolov8n.pt",
  "camera_name": "cam",
  "save_detections": true
}
```

This covers `get_detections`, `get_detections_from_camera` and
`capture_all_from_camera` — all three run through the same path.

Each call writes two files sharing a name stem:

```
2026-08-26T14-03-11-482913_a1b2c3d4.jpg
2026-08-26T14-03-11-482913_a1b2c3d4.json
```

The image is stored as the camera produced it: JPEG and PNG frames are written
byte-for-byte with no re-encoding, and anything else (RGBA, depth) is converted
to PNG so it opens in ordinary image tools. The sidecar holds the detections
plus enough context to interpret them:

```json
{
  "captured_at": "2026-08-26T14:03:11.482913+00:00",
  "service_name": "vision-1",
  "camera_name": "cam",
  "image_file": "2026-08-26T14-03-11-482913_a1b2c3d4.jpg",
  "width": 640,
  "height": 480,
  "detections": [
    {
      "confidence": 0.87,
      "class_name": "cup",
      "x_min": 12,
      "y_min": 40,
      "x_max": 96,
      "y_max": 210
    }
  ]
}
```

Frames that produced no detections are saved too, with an empty `detections`
array.

A failed write never fails the detection call — the module logs the error and
returns detections as usual.

### Syncing to the cloud

`save_dir` defaults to `yolov8/<service name>` inside viam-server's capture
directory (`$VIAM_HOME/capture`, or `~/.viam/capture` when `VIAM_HOME` is
unset). If the machine runs the **data management** service with cloud sync
enabled, it uploads every file it finds under that directory — not only the
capture files it writes itself — and **deletes each file once it has uploaded
successfully**, so the directory stays bounded on its own.

Files are written under a temporary name and renamed into place, so sync never
picks up a half-written frame.

To write somewhere else, point `save_dir` at that path — either your machine's
configured `capture_dir`, or any path listed in the data manager's
`additional_sync_paths`:

```json
{
  "model_location": "yolov8n.pt",
  "camera_name": "cam",
  "save_detections": true,
  "save_dir": "/data/yolo-captures"
}
```

Uploaded images land under **Data → Files** as ordinary files, not as a dataset
with drawn bounding boxes; the boxes stay in the JSON sidecar beside each image.

***Note:*** with no cloud sync running, nothing prunes the directory. Every
detection call writes a frame, so a continuously polled service will fill the
disk. Either enable sync, or point `save_dir` at a volume you prune yourself.

## API

The YOLOv8 resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

### get_classifications(image=*binary*)

### get_classifications_from_camera(camera_name=*string*)
