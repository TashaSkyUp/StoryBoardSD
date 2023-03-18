import base64
import random
from typing import Any

import httpx
from PIL import Image

import modules.scripts
import modules.shared as shared
from modules.processing import StableDiffusionProcessingTxt2Img, \
    process_images
from modules.shared import opts, cmd_opts
from .sb_rendering import SBMultiSampleArgs, SBImageResults
from modules.api.models import StableDiffusionTxt2ImgProcessingAPI
import requests
import json
from .env import *


def get_sd_txt_2_image_params_from_story_board_params(sb_iparams: SBMultiSampleArgs):
    # convert the story board params to StableDiffusionProcessingTxt2Img params
    if not isinstance(sb_iparams, SBMultiSampleArgs):
        raise TypeError(f"sb_iparams must be of type SBMultiSampleArgs, but is {type(sb_iparams)}")

    # get the hyper params
    prompt = sb_iparams.hyper.prompt
    negative_prompt = sb_iparams.hyper.negative_prompt
    steps = sb_iparams.hyper.steps
    seed = sb_iparams.hyper.seed
    subseed = sb_iparams.hyper.subseed
    subseed_strength = sb_iparams.hyper.subseed_strength
    cfg_scale = sb_iparams.hyper.cfg_scale

    # get the render params

    width = sb_iparams.render.width
    height = sb_iparams.render.height
    restore_faces = sb_iparams.render.restore_faces
    tiling = sb_iparams.render.tiling
    batch_count = sb_iparams.render.batch_count
    batch_size = sb_iparams.render.batch_size

    if isinstance(prompt, list):
        if batch_size > len(prompt):
            batch_size = len(prompt)
    else:
        batch_size = 1

    sampler_index = sb_iparams.render.sampler_index
    sampler_name = sb_iparams.render.sampler_name

    # convert the render params to the StableDiffusionProcessingTxt2Img params

    tmp = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=["None", "None"],
        negative_prompt=negative_prompt if type(negative_prompt) is not list else negative_prompt[0],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,  # if type(subseed_strength) is not list else subseed_strength[0],
        sampler_name=sampler_name if type(sampler_name) is not list else sampler_name[0],
        batch_size=batch_size if type(batch_size) is not list else batch_size[0],
        n_iter=1,
        steps=steps if type(steps) is not list else steps[0],
        cfg_scale=cfg_scale if type(cfg_scale) is not list else cfg_scale[0],
        width=width if type(width) is not list else width[0],
        height=height if type(height) is not list else height[0],
        restore_faces=restore_faces if type(restore_faces) is not list else restore_faces[0],
        tiling=tiling if type(tiling) is not list else tiling[0],
        seed_enable_extras=True
    )
    if isinstance(tmp.prompt, str):
        tmp.prompt = [tmp.prompt]
    return tmp


def storyboard_call_multi(params: SBMultiSampleArgs, *args, **kwargs) -> SBImageResults:
    p = get_sd_txt_2_image_params_from_story_board_params(params)

    p.scripts = modules.scripts.scripts_txt2img

    # turn all -1 seeds to random values ala modules.processing.get_fixed_seed(-1)
    for i in range(len(p.seed)):
        if p.seed[i] == -1:
            p.seed[i] = modules.processing.get_fixed_seed(-1)
    p.do_not_save_samples = True

    try:
        processed = process_images(p)
    except Exception as e:
        print(e)
        # try to process each prompt separately
        results = []
        for i in range(len(params.combined.hyper.prompt)):
            try:
                sbim = params[i]
                p = get_sd_txt_2_image_params_from_story_board_params(sbim)
                results.append(process_images(p))
            except Exception as e:
                print(e)
                results.append(None)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []
    sb_results = SBImageResults(
        processed=processed,
        # generation_info_js=generation_info_js,
        # generation_info_html=plaintext_to_html(generation_info_js)
    )

    return sb_results


async def storyboard_call_endpoint(params: SBMultiSampleArgs, *args, **kwargs) -> SBImageResults:
    p = get_sd_txt_2_image_params_from_story_board_params(params)

    p.scripts = modules.scripts.scripts_txt2img

    # turn all -1 seeds to random values ala modules.processing.get_fixed_seed(-1)
    for i in range(len(p.seed)):
        if p.seed[i] == -1:
            p.seed[i] = modules.processing.get_fixed_seed(-1)
    p.do_not_save_samples = True

    try:
        # Choose a random server URL from the list
        server_url = random.choice(STORYBOARD_RENDER_SERVER_URLS)
        processed = await call_json_api_endpoint_2(url=server_url, data=p)
    except Exception as e:
        print(e)
        # try to process each prompt separately this will fail if running UI only
        results = []
        for i in range(len(params.combined.hyper.prompt)):
            try:
                sbim = params[i]
                p = get_sd_txt_2_image_params_from_story_board_params(sbim)
                results.append(process_images(p))
            except Exception as e:
                print(e)
                results.append(None)

    shared.total_tqdm.clear()
    generation_info_js = processed["parameters"]
    images = processed["images"]
    if opts.samples_log_stdout:
        print(generation_info_js)
    processed["parameters"]["images"] = images
    if opts.do_not_show_images:
        processed.images = []
    sb_results = SBImageResults(
        api_results=processed["parameters"]
        # generation_info_js=generation_info_js,
        # generation_info_html=plaintext_to_html(generation_info_js)
    )

    return sb_results


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    from io import BytesIO
    image = Image.open(BytesIO(base64.b64decode(encoding)))
    return image


def call_json_api_endpoint(url, data):
    headers = {'content-type': 'application/json'}
    if isinstance(data, StableDiffusionProcessingTxt2Img):
        data = vars(data)
        data.pop("scripts", None)
        data.pop("script_args")
        data.pop("s_tmax")
    res = requests.post(url, data=json.dumps(data), headers=headers)
    rj = res.json()
    rj["images"] = [decode_base64_to_image(i) for i in rj["images"]]
    return rj


async def call_json_api_endpoint_2(url: str, data: Any) -> dict:
    headers = {'content-type': 'application/json'}
    if isinstance(data, StableDiffusionProcessingTxt2Img):
        data = vars(data)
        data.pop("scripts", None)
        data.pop("script_args")
        data.pop("s_tmax")

    timeout = httpx.Timeout(60.0)  # Set the read timeout to 60 seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(url=url,
                                data=json.dumps(data),
                                headers=headers)

    rj = res.json()
    rj["images"] = [decode_base64_to_image(i) for i in rj["images"]]
    return rj