import os
from os import path as osp
from lit.formats import ShTest

config.name = "MLIR-LEARN"
config.test_format = ShTest()
config.suffixes = [".mlir"]


current_path = os.getcwd()
tool_path = "build/07-varifiers/tools"

config.environment["PATH"] = (
    osp.join(current_path, tool_path) + ":" + os.environ["PATH"]
)
