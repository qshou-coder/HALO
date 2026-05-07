import os
import random
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inference.halo_inferencer import HaloInferencer


HALO_MODEL_DIR = os.environ.get("HALO_MODEL_DIR", "./downloads/pretrained_models")
HALO_CKPT_DIR = os.environ.get("HALO_CKPT_DIR", "./downloads/my_checkpoints")


class HALO:

    def __init__(self, pretrained_model_path):
        self.checkpoint_path = pretrained_model_path
        self.policy = self.make_policy(self.checkpoint_path)
        self.obs_buffer = []
        self.use_prob = 0.2

    def make_policy(self, checkpoint_path):
        model_path = os.path.join(HALO_MODEL_DIR, "BAGEL-7B-MoT")
        llm_path = os.path.join(HALO_MODEL_DIR, "Qwen_1.5B_model")
        vit_path = os.path.join(HALO_MODEL_DIR, "siglip-so400m-14-980-flash-attn2-navit")

        llm_config = Qwen2Config.from_pretrained(llm_path)
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        tokenizer = Qwen2Tokenizer.from_pretrained(llm_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vit_config = SiglipVisionConfig.from_pretrained(vit_path)
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            action_gen=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM.from_pretrained(llm_path, config=llm_config)
            vit_model = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 378, 14)

        max_mem_per_gpu = "80GiB"  # adjust to your GPU; A100-80G can host the model on a single GPU

        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device if k in device_map else "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        # Load pretrained weights
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(HALO_CKPT_DIR, "halo_pt_weight", "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload",
        )

        # Load finetuned weights
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=self.checkpoint_path,
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload",
        )
        model = model.eval()
        print('Halo loaded')

        inferencer = HaloInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        return inferencer

    def get_action(self, instruction, obs_window):
        obs_window_image = [Image.fromarray(item) for item in obs_window]

        use_prob = random.random() < self.use_prob
        cfg_action_interval = [0.4, 1.0] if use_prob else [1.0, 1.0]

        inference_hyper = dict(
            cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.4, 1.0],
            timestep_shift=3.0, num_timesteps=50,
            cfg_renorm_min=0.0, cfg_renorm_type="global",
            cfg_action_text_scale=6.0, cfg_action_img_scale=6.0,
            cfg_action_interval=cfg_action_interval,
            cfg_action_timestep_shift=3.0, cfg_action_num_timesteps=10,
            cfg_action_renorm_min=0.0, cfg_action_renorm_type="global",
            action_shape=16,
            use_subtask=use_prob, use_goal_image=use_prob,
            decode_goal_image=False,
        )

        output_dict = self.policy(image=obs_window_image, text=instruction, **inference_hyper)
        return output_dict['action']

    def update_obs(self, obs):
        self.obs_buffer.append(obs)
        return self.obs_buffer

    def reset_obs(self):
        self.obs_buffer = []
