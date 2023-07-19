import io
import cv2
import torch
import einops
import base64
import random
import numpy as np
import pandas as pd
from PIL import Image
import config as config
from annotator.hed import HEDdetector
from diffusers import DiffusionPipeline
from cldm.ddim_hacked import DDIMSampler
from diffusers import KandinskyV22Pipeline
from pytorch_lightning import seed_everything
from diffusers import KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from annotator.util import nms, resize_image, HWC3
from cldm.model import create_model, load_state_dict
from transformers import CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, DiffusionPipeline

class Generation():
    
    def __init__(self, model_dalle):
        self.model_dalle = model_dalle
        self.df = pd.read_csv("style_constructions.csv")
        
    def Gen(self, request, request_model, add_prompt, sizeH, sizeW):
    
        try:
            image_Control = request.image
            guidance_scale = request.guidance_scale
            
            images_list = []
            prompt, negative_prompt, seedVal, n_images, steps, prompt_style = request.sentence, request.negative_prompt, request.seedVal, request.n_images, request.steps, request.prompt_style
            
            negative_prompt = "low quality, bad quality" if negative_prompt == None or negative_prompt == "" else negative_prompt
            
            if seedVal == 0:
                seedVal = random.randint(1, 15000)         
            generator = torch.Generator("cuda").manual_seed(seedVal)
                
            if request_model == 'img2img':
                
                assert image_Control != None, "No image as input"
                
                input_image = self.get_input_image(image_Control)
                
                results_list = []
                num_inference_steps = steps
                
                output = self.pipe(prompt=[prompt] * n_images, 
                                   image=input_image, strength=0.75, 
                                   guidance_scale=guidance_scale, 
                                   num_inference_steps=num_inference_steps).images
                
                for img in output:
                    img = self.convert_pil_image_to_byte_array(img)
                    results_list.append(base64.b64encode(img))
                
                return results_list
                
            elif request_model == 'scribble':

                assert image_Control != None, "No image as input"
                
                input_image = np.array(self.get_input_image(image_Control))
                    
                ddim_sampler = DDIMSampler(self.pipe)
                preprocessor = None
                
                image_resolution = 768
                detect_resolution = 256
                a_prompt = 'best quality'
                guess_mode = 'Guess Mode'
                strength = 1.0
                ddim_steps = 50
                eta = 1.0
                scale = guidance_scale
                det = "Scribble_HED"
                
                if 'HED' in det:
                    if not isinstance(preprocessor, HEDdetector):
                        preprocessor = HEDdetector()

                with torch.no_grad():
                    input_image = HWC3(input_image)

                    if det == 'None':
                        detected_map = input_image.copy()
                    else:
                        detected_map = preprocessor(resize_image(input_image, detect_resolution))
                        detected_map = HWC3(detected_map)

                    img = resize_image(input_image, image_resolution)
                    H, W, C = img.shape

                    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                    detected_map = nms(detected_map, 127, 3.0)
                    detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
                    detected_map[detected_map > 4] = 255
                    detected_map[detected_map < 255] = 0

                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.stack([control for _ in range(n_images)], dim=0)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                    
                    seed_everything(seedVal)

                    if config.save_memory:
                        self.pipe.low_vram_shift(is_diffusing=False)

                    cond = {"c_concat": [control], "c_crossattn": [self.pipe.get_learned_conditioning([prompt + ', ' + a_prompt] * n_images)]}
                    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.pipe.get_learned_conditioning([negative_prompt] * n_images)]}
                    shape = (4, H // 8, W // 8)

                    if config.save_memory:
                        self.pipe.low_vram_shift(is_diffusing=True)

                    self.pipe.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

                    samples, intermediates = ddim_sampler.sample(ddim_steps, n_images,
                                                                    shape, cond, verbose=False, eta=eta,
                                                                    unconditional_guidance_scale=scale,
                                                                    unconditional_conditioning=un_cond)

                    if config.save_memory:
                        self.pipe.low_vram_shift(is_diffusing=False)

                    x_samples = self.pipe.decode_first_stage(samples)
                    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

                    results = [x_samples[i] for i in range(n_images)]
                    
                    results_list = []
                    for img in results:
                        img = Image.fromarray(img)
                        img = self.convert_pil_image_to_byte_array(img)
                        results_list.append(base64.b64encode(img))
                    return results_list
                
            elif request_model == 'dalle':
                with torch.no_grad():
                    images = self.model_dalle.generate_images(
                                    prompt, 
                                    seed=-1, 
                                    grid_size=2, 
                                    is_seamless=False,
                                    temperature=1,
                                    top_k=256, 
                                    supercondition_factor=16,
                                    is_verbose=True
                                )
                for idx, image in enumerate(images[:3]):
                    image = image.cpu().detach().numpy()
                    image = Image.fromarray((image * 1).astype(np.uint8)).convert('RGB')
                    bytes_im = self.convert_pil_image_to_byte_array(image)
                    images_list.append(base64.b64encode(bytes_im))
                    
                return images_list
            
            elif request_model == 'kandinsky2_1' or request_model == 'kandinsky2_2':
                
                all_imgs = []
                if request_model == 'kandinsky2_1':
                
                    pipe_prior = DiffusionPipeline.from_pretrained("kandinsky/kandinsky2_1-prior", torch_dtype=torch.float16, local_files_only=True)
                    pipe_prior.to("cuda")
                    
                    image_embeds, negative_image_embeds = pipe_prior(prompt, negative_prompt, guidance_scale=1.0).to_tuple()
                    
                    image = self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                        image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, 
                        height=sizeH, width=sizeW, num_images_per_prompt=n_images,
                        num_inference_steps=steps)  
                else:
                    image_encoder = CLIPVisionModelWithProjection.from_pretrained('kandinsky/kandinsky-2-2-prior', subfolder='image_encoder').half().to("cuda")
                    prior = KandinskyV22PriorPipeline.from_pretrained('kandinsky/kandinsky-2-2-prior', image_encoder=image_encoder, torch_dtype=torch.float16).to("cuda")
                    
                    img_emb = prior(
                        prompt=prompt,
                        num_inference_steps=2, 
                        num_images_per_prompt=n_images
                    )

                    negative_emb = prior(
                        prompt=negative_prompt,
                        num_inference_steps=2,
                        num_images_per_prompt=n_images,
                        guidance_scale=guidance_scale
                    )
                    
                    image = self.pipe(image_embeds=img_emb.image_embeds, 
                        negative_image_embeds=negative_emb.image_embeds, 
                        num_inference_steps=steps, 
                        height=sizeH, 
                        width=sizeW)

                for img in image[0]:
                    img = self.convert_pil_image_to_byte_array(img)
                    all_imgs.append(base64.b64encode(img))
                    
                return all_imgs

            else:
                prompt_style = prompt_style.split(",") 
                final_prompt = ""
                for pr in prompt_style: 
                    d = self.df.loc[self.df['style'] == pr]
                    if len(d) > 0:
                        d = d.reset_index()
                        text = d['prompt'][0]
                        final_prompt = final_prompt + text
                    
                final_prompt = prompt + final_prompt + add_prompt
 
                all_imgs = []
                prompt = [final_prompt] * n_images
                negative_prompt = [negative_prompt] * n_images if negative_prompt is not None else None
                images = self.pipe(prompt=prompt, negative_prompt=negative_prompt, 
                                   num_inference_steps=steps, 
                                   generator=generator, 
                                   height=sizeH, width=sizeW, 
                                   guidance_scale=guidance_scale).images
                for img in images:
                    img = self.convert_pil_image_to_byte_array(img)
                    all_imgs.append(base64.b64encode(img))

                return all_imgs
                
        except Exception as e:
            print("Error: {}".format(e))
            raise Exception("Error: {}".format(e)) 
        
    def convert_pil_image_to_byte_array(self, img):
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='JPEG', subsampling=0, quality=100)
        img_byte_array = img_byte_array.getvalue()
        return img_byte_array
    
    def get_input_image(self, image):
        input_image = Image.open(io.BytesIO(image))
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
            
        return input_image
    
    def getPath(self, request, df_models):
        request_model = request.model
        request_model_splited = request_model.split(",")
        for m in request_model_splited: 
            d = df_models.loc[df_models['model'] == m]
            assert len(d) > 0
            d = d.reset_index()
            path = d['id'][0]
            add_prompt = d['add_prompt'][0]
            #size = d['H'][0]
            
        sizeH, sizeW = int(request.H), int(request.W)
        return path, add_prompt, request_model, sizeH, sizeW
        
    def formPipe(self, path, request_model):

        if request_model == 'scribble':
            self.load_ControlNet(path)
        elif request_model == 'img2img':
            self.load_img2img(path)
        elif request_model == 'kandinsky2_1' or request_model == 'kandinsky2_2':
            self.load_kandinsky(path, request_model)
        else:
            self.load_Diffusers(path, request_model)
    
    def load_ControlNet(self, path):
        
        self.pipe = create_model(f'checkpoints/{path}.yaml').cpu()
        self.pipe.load_state_dict(load_state_dict('checkpoints/v1-5-pruned.ckpt', location='cuda'), strict=False)
        self.pipe.load_state_dict(load_state_dict('checkpoints/control_v11p_sd15_scribble.pth', location='cuda'), strict=False)
        self.pipe = self.pipe.cuda()   
        
    def load_Diffusers(self, path, request_model):
        
        if request_model == 'SD2':
            self.pipe = StableDiffusionPipeline.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16)
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        elif request_model == 'SD2_1': 
            self.pipe = StableDiffusionPipeline.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        else:
            self.pipe = StableDiffusionPipeline.from_ckpt(path, local_files_only=True, torch_dtype=torch.float16)
            
        self.pipe = self.pipe.to("cuda")
        self.pipe.vae.enable_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
    
    def load_img2img(self, path):
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_ckpt(path, local_files_only=True, torch_dtype=torch.float16).to("cuda")

        self.pipe.vae.enable_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        
    
    def load_kandinsky(self, path, request_model):
        
        if request_model == 'kandinsky2_1':
            self.pipe = DiffusionPipeline.from_pretrained(f'{path}{request_model}', torch_dtype=torch.float16, local_files_only=True).to("cuda")

        else:
            unet = UNet2DConditionModel.from_pretrained(f'{path}kandinsky-2-2-decoder', subfolder='unet').half().to("cuda")
            self.pipe = KandinskyV22Pipeline.from_pretrained(f'{path}kandinsky-2-2-decoder', unet=unet, torch_dtype=torch.float16).to("cuda")
            
