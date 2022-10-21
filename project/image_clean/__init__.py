"""Video Matte Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import todos

from . import restormer

import pdb

DERAIN_ZEROPAD_TIMES = 8


def model_forward(model, device, input_tensor, multi_times):
    # zeropad for model
    os.system("nvidia-smi | grep python")
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    output_tensor = todos.model.forward(model, device, input_tensor)

    os.system("nvidia-smi | grep python")

    return output_tensor[:, :, 0:H, 0:W]


def get_defocus_model():
    """Create model."""
    model_path = "models/image_defocus.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = restormer.Restormer()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_defocus.torch"):
        model.save("output/image_defocus.torch")

    return model, device


def get_denoise_model():
    """Create model."""
    model_path = "models/image_denoise.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = restormer.Restormer(LayerNorm_type="BiasFree")
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_denoise.torch"):
        model.save("output/image_denoise.torch")

    return model, device


def get_deblur_model():
    """Create model."""
    model_path = "models/image_deblur.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = restormer.Restormer()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_deblur.torch"):
        model.save("output/image_deblur.torch")

    return model, device


def get_derain_model():
    """Create model."""
    model_path = "models/image_derain.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    model = restormer.Restormer()
    todos.model.load(model, checkpoint, key="params")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_derain.torch"):
        model.save("output/image_derain.torch")

    return model, device


def defocus_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_defocus_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def denoise_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_denoise_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def deblur_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_deblur_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def derain_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_derain_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
