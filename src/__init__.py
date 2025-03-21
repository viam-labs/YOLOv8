"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .yolov8 import yolov8

Registry.register_resource_creator(
    Vision.API, yolov8.MODEL, ResourceCreatorRegistration(yolov8.new, yolov8.validate)
)
