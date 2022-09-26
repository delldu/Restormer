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
import time
import redos
import todos

from . import restormer

import pdb

DEFOCUS_ZEROPAD_TIMES = 8
DENOISE_ZEROPAD_TIMES = 8
DEBLUR_ZEROPAD_TIMES = 8

DERAIN_ZEROPAD_TIMES = 8

def model_load(model, model_path):
    """Create model."""

    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path
    state = torch.load(checkpoint)
    model.load_state_dict(state['params'])


def model_forward(model, device, input_tensor, multi_times):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0:H, 0:W]


def get_defocus_model():
    """Create model."""

    device = todos.model.get_device()
    model = restormer.Restormer()
    model_load(model, "models/image_defocus.pth")
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

    device = todos.model.get_device()
    model = restormer.Restormer(LayerNorm_type='BiasFree')
    model_load(model, "models/image_denoise.pth")
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

    device = todos.model.get_device()
    model = restormer.Restormer()
    model_load(model, "models/image_deblur.pth")
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

    device = todos.model.get_device()
    model = restormer.Restormer()
    model_load(model, "models/image_derain.pth")
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_derain.torch"):
        model.save("output/image_derain.torch")

    return model, device


def defocus_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.defocus(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def defocus_server(name, host="localhost", port=6379):
    # load model
    model, device = get_defocus_model()

    def do_service(input_file, output_file, targ):
        print(f"  defocus {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, DEFOCUS_ZEROPAD_TIMES)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_defocus", do_service, host, port)


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
        start_time = time.time()

        predict_tensor = model_forward(model, device, input_tensor, DEFOCUS_ZEROPAD_TIMES)

        print(f"Defocus {filename} on {device} spend {time.time() - start_time:.4f} seconds.")

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()

def denoise_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.denoise(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def denoise_server(name, host="localhost", port=6379):
    # load model
    model, device = get_denoise_model()

    def do_service(input_file, output_file, targ):
        print(f"  denoise {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, DENOISE_ZEROPAD_TIMES)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_denoise", do_service, host, port)


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
        start_time = time.time()

        predict_tensor = model_forward(model, device, input_tensor, DENOISE_ZEROPAD_TIMES)

        print(f"Denoise {filename} on {device} spend {time.time() - start_time:.4f} seconds.")
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()

def deblur_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.deblur(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def deblur_server(name, host="localhost", port=6379):
    # load model
    model, device = get_deblur_model()

    def do_service(input_file, output_file, targ):
        print(f"  deblur {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, DEBLUR_ZEROPAD_TIMES)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_deblur", do_service, host, port)


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
        start_time = time.time()

        predict_tensor = model_forward(model, device, input_tensor, DEBLUR_ZEROPAD_TIMES)

        print(f"Deblur {filename} on {device} spend {time.time() - start_time:.4f} seconds.")

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def derain_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.derain(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def derain_server(name, host="localhost", port=6379):
    # load model
    model, device = get_derain_model()

    def do_service(input_file, output_file, targ):
        print(f"  derain {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, DERAIN_ZEROPAD_TIMES)            
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_derain", do_service, host, port)


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
        start_time = time.time()

        predict_tensor = model_forward(model, device, input_tensor, DERAIN_ZEROPAD_TIMES)

        print(f"Derain {filename} on {device} spend {time.time() - start_time:.4f} seconds.")

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
