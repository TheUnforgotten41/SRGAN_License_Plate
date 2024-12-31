from typing import Any
import yaml
import cv2
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Module
from collections import OrderedDict
import model
import os
from PIL import Image

class SuperResolutionGAN:

    def __init__(self):
        #Load config
        self.sr_model = self.build_model('srresnet_x4', 'cpu')

    def process_image(self,img,range_norm,half):
        image = img.astype(np.float32) / 255.0

    # BGR image channel data to RGB image channel data
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
        if range_norm:
            tensor = tensor.mul(2.0).sub(1.0)

        # Convert torch.float32 image data type to torch.half image data type
        if half:
            tensor = tensor.half()

        return tensor
        
    def build_model(self,model_arch_name: str, device: torch.device) -> nn.Module:
        # Initialize the super-resolution model
        sr_model = model.__dict__[model_arch_name]()

        # Load model weights
        sr_model = self.load_pretrained_state_dict(sr_model,False, "g_best.pth.tar")

        # Start the verification mode of the model.
        sr_model.eval()

        # Enable half-precision inference to reduce memory usage and inference time
        # if self.config['half']:
            # sr_model.half()

        sr_model = sr_model.to(device)

        return sr_model
    
    
    def load_pretrained_state_dict(self,
            model: nn.Module,
            compile_state: bool,
            model_weights_path: str,
    ) -> Module:
        """Load pre-trained model weights

        Args:
            model (nn.Module): model
            compile_state (bool): model compilation state, `False` means not compiled, `True` means compiled
            model_weights_path (str): model weights path

        Returns:
            model (nn.Module): the model after loading the pre-trained model weights
        """

        checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["state_dict"]
        model = self.load_state_dict(model, compile_state, state_dict)
        return model
    
    @staticmethod
    def load_state_dict(
            model: nn.Module,
            compile_mode: bool,
            state_dict: dict,
    ):
        """Load model weights and parameters

        Args:
            model (nn.Module): model
            compile_mode (bool): Enable model compilation mode, `False` means not compiled, `True` means compiled
            state_dict (dict): model weights and parameters waiting to be loaded

        Returns:
            model (nn.Module): model after loading weights and parameters
        """

        # Define compilation status keywords
        compile_state = "_orig_mod"

        # Process parameter dictionary
        model_state_dict = model.state_dict()
        new_state_dict = OrderedDict()

        # Check if the model has been compiled
        for k, v in state_dict.items():
            current_compile_state = k.split(".")[0]
            if compile_mode and current_compile_state != compile_state:
                raise RuntimeError("The model is not compiled. Please use `model = torch.compile(model)`.")

            # load the model
            if compile_mode and current_compile_state != compile_state:
                name = compile_state + "." + k
            elif not compile_mode and current_compile_state == compile_state:
                name = k[10:]
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict

        # Traverse the model parameters, load the parameters in the pre-trained model into the current model
        new_state_dict = {k: v for k, v in state_dict.items() if
                        k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

        # update model parameters
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)

        return model

    @staticmethod
    def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
        """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

        Args:
            tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
            range_norm (bool): Scale [-1, 1] data to between [0, 1]
            half (bool): Whether to convert torch.float32 similarly to torch.half type.

        Returns:
            image (np.ndarray): Data types supported by PIL or OpenCV

        Examples:
            >>> example_image = cv2.imread("lr_image.bmp")
            >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

        """
        if range_norm:
            tensor = tensor.add(1.0).div(2.0)
        if half:
            tensor = tensor.half()

        image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

        return image

    def __call__(self, img):
        image = self.process_image(img,range_norm=False,half=False).unsqueeze(0)
        print(image.shape)
        # image = image.to(self.config['device'])
        with torch.no_grad():
            output = self.sr_model(image)

            

            output = output.squeeze(0)
        image = self.tensor_to_image(output,False,False)
        return image
    

if __name__ == "__main__":
    srgan = SuperResolutionGAN()
    image_path = "75-F1325.67.jpg"
    sr_image = srgan(image_path)
    img = Image.fromarray(sr_image)

    img.save("test_cpu.jpg")


