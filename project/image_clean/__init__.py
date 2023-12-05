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


def get_defocus_model():
    """Create model."""
    device = todos.model.get_device()

    model = restormer.Restormer(layer_norm_type="WithBias")
    model.load_weights("models/image_defocus.pth")
    model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_defocus.torch"):
        model.save("output/image_defocus.torch")

    return model, device

# def get_defocus2_model():
#     """Create model."""
#     device = todos.model.get_device()

#     model = restormer.Restormer(layer_norm_type="WithBias")
#     model.load_weights("models/image_defocus2.pth")
#     model = todos.model.ResizePadModel(model)
#     model = model.to(device)
#     model.eval()

#     print(f"Running on {device} ...")
#     # make sure model good for C/C++
#     model = torch.jit.script(model)
#     # https://github.com/pytorch/pytorch/issues/52286
#     torch._C._jit_set_profiling_executor(False)
#     # C++ Reference
#     # torch::jit::getProfilingMode() = false;                                                                                                             
#     # torch::jit::setTensorExprFuserEnabled(false);

#     todos.data.mkdir("output")
#     if not os.path.exists("output/image_defocus2.torch"):
#         model.save("output/image_defocus2.torch")

#     return model, device


def get_denoise_model():
    """Create model."""
    device = todos.model.get_device()
    model = restormer.Restormer(layer_norm_type="BiasFree")
    model.load_weights("models/image_denoise.pth")
    # model = todos.model.GridTileModel(model)
    model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type):
        model.float()


    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    # print(model.code)
    todos.data.mkdir("output")
    if not os.path.exists("output/image_denoise.torch"):
        model.save("output/image_denoise.torch")

    return model, device


def get_deblur_model():
    """Create model."""
    device = todos.model.get_device()
    model = restormer.Restormer()
    model.load_weights("models/image_deblur.pth")
    model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_deblur.torch"):
        model.save("output/image_deblur.torch")

    return model, device


def get_derain_model():
    """Create model."""
    device = todos.model.get_device()
    model = restormer.Restormer()
    model.load_weights("models/image_derain.pth")
    model = todos.model.ResizePadModel(model)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);
        
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

# def defocus2_predict(input_files, output_dir):
#     # Create directory to store result
#     todos.data.mkdir(output_dir)

#     # load model
#     model, device = get_defocus2_model()

#     # load files
#     image_filenames = todos.data.load_files(input_files)

#     # start predict
#     progress_bar = tqdm(total=len(image_filenames))
#     for filename in image_filenames:
#         progress_bar.update(1)

#         # orig input
#         input_tensor = todos.data.load_tensor(filename)
#         # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
#         orig_tensor = input_tensor.clone().detach()

#         predict_tensor = todos.model.forward(model, device, input_tensor)

#         output_file = f"{output_dir}/{os.path.basename(filename)}"

#         todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
#     todos.model.reset_device()



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

def denoise_predict_add_noise(input_files, output_dir):
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
        input_tensor += torch.randn(input_tensor.size()) * 25.0/255.0

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
