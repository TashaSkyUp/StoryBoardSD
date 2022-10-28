import math
import os
import sys
import traceback

import PIL.ImageEnhance
import torch
import torchvision.transforms
import torchvision.transforms as T
import numpy as np
from PIL import Image

import modules.scripts as scripts

from  modules.sd_samplers import samplers as sd_samplers
import gradio as gr
import modules.shared as shared
from modules.processing import Processed, process_images, StableDiffusionProcessing,\
    StableDiffusionProcessingTxt2Img,StableDiffusionProcessingImg2Img

from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):

    def title(self):
        return "Prompts from textbox with markdown"

    def ui(self, is_img2img):
        self.is_img2img = is_img2img
        prompt_txt = gr.TextArea(label="Prompts")
        return [prompt_txt]

    def on_show(self, checkbox_txt, file, prompt_txt):
        return [gr.TextArea.update(visible = True) ]

    def get_sampler_index(self, sampler_str: str):
        for i, sampler in enumerate(sd_samplers):
            if sampler_str == sampler[0]:
                return i

        raise ValueError(f"Sampler {sampler_str} not found.")

    def parse_generation_parameters(self, prompt_txt: str):
        # there are two sections that need to be handled in diffent ways
        # the first section is the two prompts, negative and postitive
        # they can container newlines and ":" characters as well as other special characters
        # thankfully its easy to find them at least in a non-garuanteed way


        # split the sections
        # starting at the first character ending before "Steps"

        section1 = prompt_txt[:prompt_txt.find("Steps")].strip()
        positive_prompt = section1[:section1.find("Negative"):].strip()
        negative_prompt = section1[section1.find("Negative"):].strip()
        section2 = prompt_txt[prompt_txt.find("Steps:"):].strip()

        out = {}
        out["Prompt"] = positive_prompt
        try:
            tmp = negative_prompt.split(":")[1:]
            out["Negative prompt"] = ":".join(tmp).strip()
            #out["Negative prompt"] =

        except IndexError as e:
            raise ValueError(f"Negative prompt not found in {negative_prompt},"
                             f"section1={section1},"
                             f"section2={section2}")

        # replace newlines with spaces
        section2 = section2.replace("\r", " ").replace("\n", " ")
        # replace double spaces with single spaces
        section2 = section2.replace("  ", " ")
        sec_2_csv = section2.split(",")
        for item in sec_2_csv:
            try:
                key, value = item.split(":")
            except ValueError:
                raise ValueError(f"Invalid parameter: {item}\n full section: {section2}")
            out[key.strip()] = value.strip()


        return out

    def run(self, p: StableDiffusionProcessingTxt2Img, prompt_txt: str):

        zoom_percent = 98
        noise_percent = 33
        denoise_strength = 0.70

        image_param_data = [x for x in prompt_txt.split("#")]

        image_param_data = [x for x in image_param_data if len(x) > 2]

        img_count = len(image_param_data)# * p.n_iter
        #batch_count = math.ceil(img_count / p.batch_size)
        #loop_count = math.ceil(batch_count / p.n_iter)
        print(f"Will process {img_count} images.")

        p.do_not_save_grid = False

        state.job_count = img_count

        images = []
        start_images = []

        for loop_no,param_data in enumerate(image_param_data):
            # update webui state
            state.job = f"{loop_no + 1} out of {img_count}"

            # parse parameters
            single_prompt_dict = self.parse_generation_parameters(param_data)
            new_p = StableDiffusionProcessing()

            # set the individual parameters
            p.sampler_index = self.get_sampler_index(single_prompt_dict["Sampler"])
            p.prompt = single_prompt_dict["Prompt"]
            p.negative_prompt = single_prompt_dict["Negative prompt"]
            p.width = int(single_prompt_dict["Size"].split("x")[0])
            p.height = int(single_prompt_dict["Size"].split("x")[1])
            p.seed = int(single_prompt_dict["Seed"])
            p.cfg_scale = float(single_prompt_dict["CFG scale"])
            p.steps = int(single_prompt_dict["Steps"])

            print(f"info: {p.__repr__()}")
            # process the image

            if loop_no > 0:

                zoom_size_w = p.width  * zoom_percent // 100
                zoom_size_h = p.height * zoom_percent // 100
                # if we are not img2img and we are not the first image
                # then we need to use the previous image

                mod_image = images[-1]

                mod_image = T.ToTensor()(mod_image)

                # fix colors
                # normalize the image
                mod_image = (mod_image - mod_image.min()) / (mod_image.max() - mod_image.min())

                # apply reference_range
                mod_image = mod_image * reference_range

                # apply reference_min
                mod_image = mod_image + reference_min

                # convert back to pil image
                mod_image = T.ToPILImage()(mod_image)

                # decontrast
                mod_image= PIL.ImageEnhance.Contrast(mod_image).enhance(0.6)


                #translate the image with PIL
                mod_image = mod_image.transform((p.width, p.height), Image.AFFINE, (1, 0, 10, 0, 1, 0))

                #back to tensor
                mod_image = T.ToTensor()(mod_image)

                # add noise
                gen = torch.Generator()
                torch.manual_seed(p.seed+loop_no)
                noise = torch.rand_like(mod_image)

                mod_image =  (noise * (noise_percent/100)) + (mod_image * (1 - (noise_percent/100)))

                # zoom in on the center of the image
                mod_image = mod_image[:,
                            p.height//2-zoom_size_h//2:p.height//2+zoom_size_h//2,
                            p.width//2-zoom_size_w//2:p.width//2+zoom_size_w//2]


                # resize the image
                mod_image = T.Resize((p.height, p.width))(mod_image)

                # convert back to pil image
                mod_image = T.ToPILImage()(mod_image)

                start_images.append(mod_image)
                #mod_image = np. images[-1] * 0.5 + np.random.rand(p.height, p.width, 3) * 0.5

                p_for_img_to_img = StableDiffusionProcessingImg2Img([mod_image])

                p_for_img_to_img.prompt = p.prompt
                p_for_img_to_img.negative_prompt = p.negative_prompt
                p_for_img_to_img.sampler_index = p.sampler_index
                p_for_img_to_img.width = p.width
                p_for_img_to_img.height = p.height
                p_for_img_to_img.seed = p.seed
                p_for_img_to_img.cfg_scale = p.cfg_scale
                p_for_img_to_img.steps = p.steps
                p_for_img_to_img.seed = p.seed
                p_for_img_to_img.sd_model = p.sd_model
                p_for_img_to_img.n_iter = p.n_iter
                p_for_img_to_img.batch_size = p.batch_size
                p_for_img_to_img.do_not_save_grid = p.do_not_save_grid
                #p_for_img_to_img.do_not_save_image = p.do_not_save_image
                #p_for_img_to_img.do_not_save_video = p.do_not_save_video
                p_for_img_to_img.restore_faces = p.restore_faces
                p_for_img_to_img.color_corrections  = p.color_corrections
                p_for_img_to_img.overlay_images = p.overlay_images
                p_for_img_to_img.outpath_samples = p.outpath_samples
                p_for_img_to_img.denoising_strength = denoise_strength



                proc = process_images(p_for_img_to_img)
            else:
                proc = process_images(p)
                reference_image = proc.images[0]
                reference_image_torch  = torchvision.transforms.functional.to_tensor(reference_image)
                reference_min = reference_image_torch.min()
                reference_max = reference_image_torch.max()
                reference_range = reference_max - reference_min
                del reference_image_torch


            # add the image to the list for the webui
            images += proc.images

        images += start_images
        return Processed(p, images, p.seed, "")
