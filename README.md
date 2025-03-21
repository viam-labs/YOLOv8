# YOLOv8 modular service

This module implements the [RDK vision API](https://github.com/rdk/vision-api) in a viam-labs:vision:yolov8 model.

This model leverages the [Ultralytics inference library](https://docs.ultralytics.com/) to allow for object detection and classification from YOLOv8 models.

Both locally deployed YOLOv8 models and models from web sources like [HuggingFace](https://huggingface.co/models?other=yolov8) can be used (HuggingFace models will be downloaded and used locally).

![Example screen recording of usage](./demo.gif)

> [!NOTE]
> Before configuring your vision service, you must [create a machine](https://docs.viam.com/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `vision` type, then select the `viam-labs:vision:yolov8` model.
Enter a name for your vision and click **Create**.

## Configure your vision service

Copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "model_location": "<string>"
}
```

> [!NOTE]
> For more information, see [Configure a Robot](https://docs.viam.com/build/configure/).

### Attributes

The following attributes are available for `viam-labs:vision:yolov8` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_location` | string | **Required** |  YOLO model name (such as "yolov8n.pt"), local path to model, or HuggingFace model repo identifier |
| `model_name` | string | Optional |  Name of model file when using HuggingFace repo identifier as `model_location` |
| `task` | string | Optional |  Name of computer vision task performed by the model: "detect" (default) or "classify" |

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

## API

The YOLOv8 resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Note: if using this method, any cameras you are using must be set in the `depends_on` array for the service configuration, for example:

```json
      "depends_on": [
        "cam"
      ]
```

### get_classifications(image=*binary*)

### get_classifications_from_camera(camera_name=*string*)

Note: if using this method, any cameras you are using must be set in the `depends_on` array for the service configuration, for example:

```json
      "depends_on": [
        "cam"
      ]
```
