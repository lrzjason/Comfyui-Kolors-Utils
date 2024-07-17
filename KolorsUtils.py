import json
import os
import random
import folder_paths
import comfy.sd
import comfy.utils
import torch
from comfy import model_management
import safetensors


class SaveWeightAsKolorsUnet:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": { 
                        "model": ("MODEL",),
                        "filename": ("STRING", {"default": "checkpoints/ComfyUI"}),
                    },
                }
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "KolorsUtils/model_merging"

    def save(self, model, filename="checkpoints/ComfyUI"):
        kolors_keys_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"kolors_keys.json")
        kolors_keys = []
        with open(kolors_keys_path, 'r', encoding='utf-8') as file:
            kolors_keys = json.load(file)
        if not os.path.exists(kolors_keys_path):
            print("KolorsKeys.json not found, please download it from github")
            raise Exception("KolorsKeys.json not found, please download it from github")
        if len(kolors_keys) == 0:
            print("KolorsKeys.json is empty, please download it from github")
            raise Exception("KolorsKeys.json is empty, please download it from github")
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename, self.output_dir)
        output_checkpoint = f"{filename}.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        print("save checkpoint to:",output_checkpoint)
        load_models = [model]
        model_management.load_models_gpu(load_models, force_patch_weights=True)
        sd = model.model.state_dict_for_saving(None, None, None)

        for k in sd:
            t = sd[k]
            if not t.is_contiguous():
                sd[k] = t.contiguous()

        Kolors = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 5632, 'dtype': torch.float16, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

        mapping = comfy.utils.unet_to_diffusers(Kolors)
        prefix = "model.diffusion_model."
        missing_tensors_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"missing_tensors.safetensors")
        missing_tensors_weight = safetensors.safe_open(missing_tensors_path, 'pt')
        ori_keys = missing_tensors_weight.keys()
        new_diffusers_weight = {key:missing_tensors_weight.get_tensor(key) for key in ori_keys}
        print("convert begin")
        err_k = ""
        err_v = ""
        for k, v in mapping.items():
            if k not in kolors_keys:
                print(k,"not in ori_keys")
                continue
            try:
                err_k = k
                err_v = v
                diffusion_model_key = f"{prefix}{v}"
                model_value = sd[diffusion_model_key]
                new_diffusers_weight[k] = model_value
            except:
                print("convert error")
                print(err_k,err_v)
        comfy.utils.save_torch_file(new_diffusers_weight, output_checkpoint, metadata={'format': 'pt'})
        
        return {}

NODE_CLASS_MAPPINGS = {
    "Save Weight As Kolors Unet": SaveWeightAsKolorsUnet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KolorsUtils": "SaveWeightAsKolorsUnet",
}