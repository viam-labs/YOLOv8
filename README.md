# YOLOv8 modular service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a viam-labs:vision:yolov8 model.

This model leverages the [Ultralytics inference library](https://docs.ultralytics.com/) to allow for object detection and classification from YOLOv8 models.

Both locally deployed YOLOv8 models and models from web sources like [HuggingFace](https://huggingface.co/models?other=yolov8) can be used (HuggingFace models will be downloaded and used locally).

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam-labs:vision:yolov8` model from the [viam-labs YOLOv8 module](https://app.viam.com/module/viam-labs/yolov8).

## Configure your vision

> [!NOTE]  
> Before configuring your vision, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `vision` type, then select the `viam-labs:vision:yolov8` model.
Enter a name for your vision and click **Create**.

On the new component panel, copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "model_location": "<model_path>"
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam-labs:vision:yolov8` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_location` | string | **Required** |  Local path or HuggingFace model identifier |

### Example Configurations

[HuggingFace model](https://huggingface.co/keremberke/yolov8n-hard-hat-detection):

```json
{
  "model_location": "keremberke/yolov8n-hard-hat-detection"
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
