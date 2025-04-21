import asyncio
from viam.module.module import Module
try:
    from models.ncnn import Ncnn
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.ncnn import Ncnn


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
