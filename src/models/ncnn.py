from pathlib import Path
from typing import ClassVar, Dict, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.utils import ValueTypes, dict_to_struct, struct_to_dict

import cv2
import ncnn
from ncnn.model_zoo import get_model
# from ncnn.utils import print_topk


class Ncnn(MLModel, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("hipsterbrown", "mlmodel"), "ncnn")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this MLModel service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any implicit dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Sequence[str]: A list of implicit dependencies
        """
        attrs = struct_to_dict(config.attributes)
        model_path = attrs.get("model_path")
        model_name = attrs.get("model_name")
        if not model_path and not model_name:
            raise Exception(
                "model_path or model_name must be specified. model is required for ncnn mlmodel service."
            )
        return []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both implicit and explicit)
        """
        attrs = struct_to_dict(config.attributes)
        self.model_path = str(attrs.get("model_path", ""))
        self.model_name = str(attrs.get("model_name", ""))
        self.label_path = str(attrs.get("label_path", ""))
        self.use_gpu = bool(attrs.get("use_gpu", False))
        self.num_threads = int(attrs.get("num_threads", 2))

        if self.model_name != "":
            self.net = get_model(
                self.model_name, num_threads=self.num_threads, use_gpu=self.use_gpu
            )

        if self.model_path != "":
            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = self.use_gpu
            self.net.opt.num_threads = self.num_threads

            weights = Path(self.model_path)
            if not weights.is_file():
                weights = next(weights.glob("*.param"))
            self.net.load_param(str(weights))
            self.net.load_model(str(weights.with_suffix(".bin")))

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, NDArray]:
        cv_img = self.process_input(input_tensors)
        results = self.net(cv_img)

        return self.process_output(results)

    def process_input(self, input_tensors: Dict[str, NDArray]):
        image_tensor = input_tensors.get("image")
        if image_tensor is None:
            raise Exception("Missing image tensor")
        image_tensor = image_tensor[0]

        image_shape = image_tensor.shape

        if len(image_shape) == 2:
            cv_img = cv2.cvtColor(image_tensor.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        elif len(image_shape) == 3 and image_shape[2] == 1:
            cv_img = cv2.cvtColor(
                image_tensor.squeeze().astype(np.uint8), cv2.COLOR_GRAY2BGR
            )

        elif len(image_shape) == 3 and image_shape[2] == 3:
            if image_tensor.dtype != np.uint8:
                if image_tensor.max() <= 1.0:
                    image_tensor = (image_tensor * 255).astype(np.uint8)
                else:
                    image_tensor = image_tensor.astype(np.uint8)

            cv_img = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2BGR)
        elif len(image_shape) == 4 and image_shape[2] == 4:
            if image_tensor.dtype != np.uint8:
                image_tensor = (image_tensor * 255).astype(np.uint8)

            cv_img = cv2.cvtColor(image_tensor, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported numpy array shape: {image_shape}")

        return cv_img

    def process_output(self, results):
        topk = 5
        indexes = np.argsort(results)[::-1][0:topk]
        scores = results[indexes]
        return {"probability": scores}

    async def metadata(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Metadata:
        target_size = self.net.target_size if self.model_name else -1

        input_info, output_info = [], []

        input_shapes, output_shapes = (
            {"image": (1, target_size, target_size, 3)},
            {"probability": (1, 1000)},
        )

        """detections output shape
        {
            "Location": (1, 100, 4),
            "Category": (1, 100),
            "Score": (1, 100)
        }
        """

        for input_tensor_name, shape in input_shapes.items():
            input_info.append(
                TensorInfo(name=input_tensor_name, shape=shape, data_type="uint8")
            )

        for output_tensor_name, shape in output_shapes.items():
            output_info.append(
                TensorInfo(
                    name=output_tensor_name,
                    shape=shape,
                    data_type="uint8",
                    extra=dict_to_struct({"label": self.label_path})
                    if self.label_path
                    else None,
                )
            )

        return Metadata(name="ncnn", input_info=input_info, output_info=output_info)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()
