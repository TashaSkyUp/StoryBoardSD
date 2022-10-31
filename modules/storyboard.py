import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.story_squad import CallArgsAsData


def storyboard(call_args_data: CallArgsAsData, *args):
    from modules.ui import plaintext_to_html
    print("Processing...")

    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=call_args_data.prompt,
        styles=["None", "None"],
        negative_prompt=call_args_data.negative_prompt,
        seed=call_args_data.seed,
        subseed=call_args_data.subseed,
        subseed_strength=call_args_data.subseed_strength,
        sampler_index=call_args_data.sampler_index,
        batch_size=1,
        n_iter=1,
        steps=call_args_data.steps,
        cfg_scale=call_args_data.cfg_scale,
        width=call_args_data.width,
        height=call_args_data.height,
        restore_faces=call_args_data.restore_faces,
        tiling=call_args_data.tiling
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\nStoryBoard: {call_args_data.prompt}", file=shared.progress_print_out)

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

    return processed.images, processed.seed, generation_info_js, plaintext_to_html(processed.info)
