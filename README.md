# Module ncnn-ml 

Use the [ncnn](https://github.com/Tencent/ncnn/) neural network inference framework to run computer vision models optimized for mobile runtimes, such as the Raspberry Pi, NVIDIA Jetson, Android, iOS, and more.

## Model hipsterbrown:mlmodel:ncnn

Use the [ncnn](https://github.com/Tencent/ncnn/) neural network inference framework to run computer vision models

### Configuration
The following attribute template can be used to configure this model:

```json
{
    "model_name": <string>,
    "use_gpu": <boolean>,
    "num_threads": <integer>
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `model_name` | string  | Required | Name of a pre-trained model from the [ncnn model zoo](https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo/model_zoo.py#L37-L58) supported by this module |
| `use_gpu` | boolean  | Optional | Set this to `true` to enable the Vulkan-based GPU inferencing on the target device. |
| `num_threads` | integer  | Optional | The number of CPU threads to use while running this service, Defaults to 2. |

**Supported Models:**
- `squeezenet_ssd` (object detection)
- `mobilenetv2_ssdlite` (object detection)
- `mobilenetv3_ssdlite` (object detection)
- `mobilenetv2_yolov3` (object detection)
- `yolov4_tiny` (object detection)
- `yolov4` (object detection)

#### Example Configuration

```json
{
  "model_name": "squeezenet_ssd",
  "num_threads": 4
}
```

## Development

If there is a missing feature that you would like to see in this module, please create a GitHub issue with your use case.
