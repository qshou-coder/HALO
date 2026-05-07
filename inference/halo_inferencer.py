from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from inference.inferencer import InterleaveInferencer


class HaloInferencer(InterleaveInferencer):
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        super().__init__(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)

    @torch.no_grad()
    def halo_inference(
        self,
        input_list: List[Union[Image.Image, str]],
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_action_text_scale: float = 4.0,
        cfg_action_img_scale: float = 2.0,
        cfg_action_interval: List[float] = [0.2, 1.0],
        cfg_action_timestep_shift: float = 5.0,
        cfg_action_num_timesteps: int = 50,
        cfg_action_renorm_min: float = 0.0,
        cfg_action_renorm_type: str = "global",
        max_think_token_n: int = 1000,
        image_shape: tuple = (1024, 1024),
        action_shape: int = 16,
        use_subtask: bool = False,
        use_goal_image: bool = False,
        decode_goal_image: bool = True,
    ) -> List[Union[str, Image.Image]]:

        output_history = []
        generated_text = None

        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for item in input_list:
                if isinstance(item, Image.Image):
                    processed_image = item
                    if use_goal_image:
                        processed_image = self.vae_transform.resize_transform(pil_img2rgb(item))
                    processed_image_shape = processed_image.size[::-1]
                    gen_context = self.update_context_image(processed_image, gen_context, vit=True, vae=use_goal_image)
                    cfg_text_context = self.update_context_image(item, gen_context, vit=True, vae=True)
                elif isinstance(item, str):
                    cfg_img_context = self.update_context_text(item, cfg_img_context)
                    gen_context = self.update_context_text(item, gen_context)
                else:
                    raise ValueError(f"Unknown input type: {type(item)}")

            ### --- start inference --- ###
            # First, generate textual subtasks...
            if use_subtask:
                generated_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_history.append(generated_text)
                print(f"Generated text: {generated_text}")

                gen_context = self.update_context_text(generated_text, gen_context)

            # Next, generate goal images...
            if use_goal_image:
                goal_output = self.gen_image(
                    processed_image_shape,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    output_type="image" if decode_goal_image else "latent",
                )

                if decode_goal_image:
                    generated_image = goal_output
                    output_history.append(generated_image)

                    processed_gen_image = pil_img2rgb(generated_image)
                    processed_gen_image = self.vae_transform.resize_transform(pil_img2rgb(processed_gen_image))
                    gen_context = self.update_context_image(processed_gen_image, gen_context, vit=True, vae=True)
                else:
                    generated_latent = goal_output
                    H, W = processed_image_shape
                    latent_h = H // self.model.latent_downsample
                    latent_w = W // self.model.latent_downsample
                    gen_context = self.update_context_with_latents(generated_latent, [(latent_h, latent_w)], gen_context)

            generated_action = self.gen_action(
                action_shape,
                gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_action_text_scale,
                cfg_img_scale=cfg_action_img_scale,
                cfg_interval=cfg_action_interval,
                timestep_shift=cfg_action_timestep_shift,
                num_timesteps=10,
                cfg_renorm_min=cfg_action_renorm_min,
                cfg_renorm_type=cfg_action_renorm_type,
            )
            output_history.append(generated_action)

        return output_history

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None, 'action': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, list):
                for item in image:
                    input_list.append(item)
            else:
                input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.halo_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
            elif isinstance(i, tuple):
                output_dict['action'] = i[0].tolist()
        return output_dict
