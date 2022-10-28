import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
#from modules.ui import plaintext_to_html
from story_squad import CallArgsAsData
from ui import plaintext_to_html
#from modules.story_squad import CallArgsAsData


#def storyboard(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, scale_latent: bool, denoising_strength: float, *args):

def storyboard(call_args_data:CallArgsAsData, *args):
    print("Processing...")
    #return
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=call_args_data.prompt,
        styles=["",""],
        negative_prompt=call_args_data.negative_prompt,
        seed=call_args_data.seed,
        subseed=call_args_data.subseed,
        subseed_strength=call_args_data.subseed_strength,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=False,
        sampler_index=call_args_data.sampler_index,
        n_iter=1,
        batch_size=1,
        steps=call_args_data.steps,
        cfg_scale=call_args_data.cfg_scale,
        width=call_args_data.width,
        height=call_args_data.height,
        restore_faces=call_args_data.restore_faces,
        tiling=call_args_data.tiling,
        enable_hr=call_args_data.enable_hr,
        scale_latent=call_args_data.scale_latent if call_args_data.enable_hr else None,
        denoising_strength=call_args_data.denoising_strength if call_args_data.enable_hr else None,
    )
    print("Processing...")
    if cmd_opts.enable_console_prompts:
        print(f"\nstoryboard: {call_args_data.prompt}", file=shared.progress_print_out)

    # check if args[0] is a tuple
    if isinstance(args[0], tuple):
        args = args[0]
    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    #return processed.images,plaintext_to_html(processed.info)
    return processed.images,processed.seed, generation_info_js, plaintext_to_html(processed.info)

