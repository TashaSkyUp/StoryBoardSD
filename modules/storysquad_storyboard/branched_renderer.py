import asyncio
import random
import logging
import time
from PIL import Image
from typing import Callable, Any
from modules.storysquad_storyboard.sb_rendering import SBImageResults
import numpy as np
from modules.storysquad_storyboard.constants import *
from modules.storysquad_storyboard.env import *

logging.basicConfig(filename="branched_renderer.log", level=1)
logger: logging.Logger = logging.getLogger("BranchedRenderer")
logger.setLevel(logging.DEBUG)
logger.info("entering branched renderer")

from modules.storysquad_storyboard.sb_rendering import DefaultRender, create_voice_over_for_storyboard, \
    SBMultiSampleArgs, save_tmp_images_for_debug, get_frame_deltas, MAX_BATCH_SIZE, get_linear_interpolation, \
    get_img_diff, compose_file_handling

from modules.storysquad_storyboard.storyboard import StoryBoardPrompt, StoryBoardSeed, StoryBoardData, SBIHyperParams


def sync_renderer(minimum_via_diff,
                  num_frames,
                  num_keyframes,
                  sb_rend_func,
                  sb_prompt,
                  sb_seeds,
                  seconds,
                  ui_params,
                  my_ren_p, ):
    # sb_prompt_first = sb_prompt[0.0:sb_prompt.total_seconds / 2]
    # sb_prompt_second = sb_prompt[sb_prompt.total_seconds / 2:sb_prompt.total_seconds]

    async def async_renderer_wrapper():
        n = len(STORYBOARD_RENDER_SERVER_URLS)
        n = 8
        duration_per_slice = seconds / n
        num_frames_per_slice = int(num_frames / n)

        awaitables = []
        for i in range(n):
            start_time = i * duration_per_slice
            end_time = (i + 1) * duration_per_slice
            sb_slice = sb_prompt[start_time:end_time]
            sb_seeds_slice = sb_seeds[start_time:end_time]
            rend_func = get_rend_func_async(sb_slice,
                                            sb_seeds_slice,
                                            ui_params,
                                            my_ren_p,
                                            sb_rend_func)
            awaitable = renderer(minimum_via_diff,
                                 num_frames_per_slice,
                                 num_keyframes,
                                 rend_func,
                                 duration_per_slice,
                                 part=i)
            awaitables.append(awaitable)

        results = await asyncio.gather(*awaitables)
        return results

    try:
        gr_loop = asyncio.get_running_loop()
    except RuntimeError as e:
        gr_loop = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_renderer_wrapper())
    loop.close()
    return result


async def renderer(minimum_via_diff: float,
                   num_frames: int,
                   num_keyframes: int,
                   rend_func: callable,
                   seconds_length: float,
                   part=0) -> [Image]:
    """
    Renders a sequence of images using the given rendering function.

    Args:
        minimum_via_diff (float): The minimum mean difference required to generate new frames.
        num_frames (int): The total number of frames to generate.
        num_keyframes (int): The number of keyframes to use when generating the image sequence.
        rend_func (callable): The rendering function to use to generate the images.
        sb_prompt (str): The text to display as a prompt when rendering each image.
        seconds_length (float): The duration of the sequence in seconds.

    Returns:
        List[Image]: A list of the rendered images.

    Raises:
        ValueError: If any of the arguments are invalid or the rendering function fails.
    """
    # create random name for renderer
    renderer_name = str(random.randint(1000, 9999))
    # seed the image list with a batch of images
    all_imgs_by_seconds_times = [i for i in np.linspace(part * seconds_length, seconds_length + (seconds_length * part),
                                                        MAX_BATCH_SIZE)]
    result = await rend_func(all_imgs_by_seconds_times)
    all_imgs_by_seconds_images = result.all_images[-MAX_BATCH_SIZE:]
    all_imgs_by_seconds = dict(zip(all_imgs_by_seconds_times, all_imgs_by_seconds_images))
    # set the first batch of images to be keyframes
    for i in all_imgs_by_seconds:
        all_imgs_by_seconds[i].info[f"frame_type"] = "k"
    done_pairs = []
    loop_count = -1
    while True:
        loop_count += 1
        all_imgs_by_seconds = dict(sorted(all_imgs_by_seconds.items()))

        imgs_by_seconds_keys_list = list(all_imgs_by_seconds.keys())
        # find the largest difference in means between two records
        frame_deltas = get_frame_deltas(list(all_imgs_by_seconds.values()))

        # construct the list of candidate pairs
        candidate_img_pairs_by_diff = []  # [delta, seconds_idx_1, seconds_idx_2]
        for delta_idx in range(len(frame_deltas)):
            time_pair = (imgs_by_seconds_keys_list[delta_idx], imgs_by_seconds_keys_list[delta_idx + 1])

            if time_pair not in done_pairs:
                candidate_img_pairs_by_diff.append(
                    (frame_deltas[delta_idx],
                     (imgs_by_seconds_keys_list[delta_idx],
                      imgs_by_seconds_keys_list[delta_idx + 1])
                     )
                )
            else:
                logger.info(f"{renderer_name}: skipping {time_pair} because it was already done")

        imgs_pairs_by_diff_sorted = sorted(candidate_img_pairs_by_diff, key=lambda x: x[0], reverse=True)

        maxdiff = imgs_pairs_by_diff_sorted[0][0]
        logger.info(f"{renderer_name}: worst pair diff: {maxdiff},mean: {np.mean(frame_deltas)}")

        # if the largest difference is less than some value then we are done
        # if we have rendered enough images, then stop
        if maxdiff < minimum_via_diff:
            if maxdiff < (minimum_via_diff * .1):
                if len(all_imgs_by_seconds) >= int(num_frames * .9):
                    logger.info(f"{renderer_name}: done because of low maxdiff")
                    break

        if len(all_imgs_by_seconds) >= int(num_frames * 1.1):
            logger.info(f"{renderer_name}: done because of too many frames")
            break

        # get a batch of the worst images
        worst_pairs_batch = imgs_pairs_by_diff_sorted[:MAX_BATCH_SIZE]
        # remove any pairs that are less than the minimum via diff
        worst_pairs_batch = [v for v in worst_pairs_batch if v[0] >= minimum_via_diff]

        # remove any pairs that have an i type frame
        worst_pairs_gen_batch = [v for v in worst_pairs_batch if
                                 "i" not in all_imgs_by_seconds[v[1][0]].info["frame_type"] and
                                 "i" not in all_imgs_by_seconds[v[1][1]].info["frame_type"]]

        # we need the list of worst pairs with i types to interpolate
        worst_pairs_int_batch = [v for v in worst_pairs_batch if
                                 "i" in all_imgs_by_seconds[v[1][0]].info["frame_type"] or
                                 "i" in all_imgs_by_seconds[v[1][1]].info["frame_type"]]
        del worst_pairs_batch

        # if there are no pairs to generate, then we need to look at if we have enough frames to be done
        if len(worst_pairs_gen_batch) == 0 and len(worst_pairs_int_batch) == 0:
            if len(all_imgs_by_seconds) >= int(num_frames * .9):
                logger.info(f"{renderer_name}: done due to nothing to do")
                break
            else:
                # this is not necessarily necessary...up to Product team.
                # this means that we are not done, but we don't have enough frames to be done,
                # so we need to lower the minimum via diff to get more frames
                logger.info(f"{renderer_name}: lowing target maxdiff because of nothing to do")
                minimum_via_diff = minimum_via_diff * .9
                logger.info(f"{renderer_name}: new minimum_via_diff: {minimum_via_diff} old: {minimum_via_diff * 1.1}")
                continue

        # update done_pairs
        pairs_to_gen = [v[1] for v in worst_pairs_gen_batch]
        done_pairs.append(pairs_to_gen)

        # get the target times for each pair
        target_times = [np.mean(v[1:]) for v in worst_pairs_gen_batch]

        # render the batch
        if len(target_times) > 0:
            imgs = await rend_func(target_times)
            imgs = imgs.all_images
            imgs = imgs[-len(target_times):] or [imgs[0]]
            pairs_done_this_iter = pairs_to_gen
            del pairs_to_gen

        # interpolate the i type frames
        for pair in worst_pairs_int_batch:
            # get the two images
            imga = all_imgs_by_seconds[pair[1][0]]
            imgb = all_imgs_by_seconds[pair[1][1]]
            # get the time of the frame
            time_of_segment = pair[1][1] - pair[1][0]
            segmnt_per_of_time = time_of_segment / seconds_length
            # get the time of the frame
            time_of_frame = pair[1][0] + (pair[0] * segmnt_per_of_time)
            # interpolate
            interp_img = get_linear_interpolation(imga, imgb, .5)
            # add to the list of images
            imgs.append(interp_img)
            # add to the list of pairs
            pairs_done_this_iter.append(pair[1])
            # add to the list of target times
            target_times.append(time_of_frame)
            # add to the list of done pairs
            done_pairs.append(pair[1])

        tt_imgs = zip(target_times, pairs_done_this_iter, imgs)
        # handle the incomming frames
        for t_of_f, src_pair, g_img in tt_imgs:
            imga = all_imgs_by_seconds[src_pair[0]]
            imgb = all_imgs_by_seconds[src_pair[1]]
            time_of_segment = src_pair[1] - src_pair[0]
            # segmnt_per_of_time = time_of_segment / seconds_length

            diff_a = get_img_diff(imga, g_img)
            diff_b = get_img_diff(g_img, imgb)
            # diff_mean = (diff_a + diff_b) / 2

            # this is the difference between the two images
            # TODO: this maybe can be safely retrieved from frame_deltas
            diff_orig = get_img_diff(imga, imgb)

            if diff_a <= 0.0002 or diff_b <= 0.0002:
                logger.info(f"{renderer_name}:frame {t_of_f} from pair {src_pair} must be i frame because a diff is 0")
                frame_type = "i"
            elif len(all_imgs_by_seconds) < num_keyframes:
                frame_type = "keyframe"
            # elif time_of_segment < (1 / my_ren_p.fps):  # if the segment is less than a frame
            #    frame_type = "i"
            #    logger.info(f'frame {t_of_f} from pair {src_pair} must be i frame because segment is less than a frame')
            elif diff_a <= diff_orig and diff_b < diff_orig:  # if the difference is less than the original difference
                frame_type = "g"
                logger.info(
                    f"{renderer_name}:frame {t_of_f} from pair {src_pair} must be g frame because a diff is less than orig")
            else:  # if the difference is greater than the original difference
                frame_type = "i"

            if frame_type == "keyframe" or frame_type == "g":
                logger.info(f"{renderer_name}:adding -{frame_type}- frame @ time: {t_of_f} for pair: {src_pair}")
                g_img.info["frame_type"] = "g"
                all_imgs_by_seconds[t_of_f] = g_img

            elif frame_type == "i":
                i_frames_needed = round(diff_orig / 3) + 1  # this is a constant I came to by trail and error
                i_frames_needed = min(i_frames_needed, 1)
                # the above actually needs to be changed because the difference score is now from 0 to 1 instead of 0
                # to 255 and is no longer linear

                logger.info(f'adding {i_frames_needed} -{frame_type}'
                            f'- (interpolation) frames between {src_pair[0]} and {src_pair[1]}')

                p_s = np.interp(fp=[0, 1],
                                xp=[0, 1],
                                x=[float(i) for i in np.arange(1 / (1 + i_frames_needed),
                                                               0.9999999999,
                                                               1 / (1 + i_frames_needed))
                                   ])

                for p in p_s:
                    k = p * time_of_segment + src_pair[0]
                    v = get_linear_interpolation(imga, imgb, p)
                    v.info["frame_type"] = "i"
                    all_imgs_by_seconds[k] = v

    return part, all_imgs_by_seconds


def get_rend_func(sb_prompt: StoryBoardPrompt, sb_seeds: StoryBoardSeed, ui_params: [], my_ren_p, render_func):
    """
    This function composes the render function that will be used to call the render function
    """
    get_sbih_for_time: Callable[[Any], SBIHyperParams] = lambda time_idx: \
        SBIHyperParams(prompt=sb_prompt[time_idx],
                       negative_prompt=ui_params[1],
                       steps=ui_params[2],
                       seed=sb_seeds.get_prime_seeds_at_times(time_idx),
                       subseed=sb_seeds.get_subseeds_at_times(time_idx),
                       subseed_strength=sb_seeds.get_subseed_strength_at_times(time_idx),
                       cfg_scale=ui_params[13]
                       )
    render_time_idx: Callable[[Any], Any] = lambda time_idx: render_func(
        SBMultiSampleArgs(hyper=get_sbih_for_time(time_idx),
                          render=my_ren_p))
    return render_time_idx


def get_rend_func_async(sb_prompt: StoryBoardPrompt, sb_seeds: StoryBoardSeed, ui_params: [], my_ren_p, render_func):
    """
    This function composes the render function that will be used to call the render function
    """

    async def render_time_idx(time_idx: Any) -> Any:
        new_hyper = SBIHyperParams(prompt=sb_prompt[time_idx],
                                   negative_prompt=ui_params[1],
                                   steps=ui_params[2],
                                   seed=sb_seeds.get_prime_seeds_at_times(time_idx),
                                   subseed=sb_seeds.get_subseeds_at_times(time_idx),
                                   subseed_strength=sb_seeds.get_subseed_strength_at_times(time_idx),
                                   cfg_scale=ui_params[13]
                                   )
        sb_multi_sample_args = SBMultiSampleArgs(
            hyper=new_hyper,
            render=my_ren_p
        )
        return await render_func(sb_multi_sample_args)

    return render_time_idx


def do_compose_setup(my_ren_p,
                     ui_params,
                     storyboard_params):
    """
    This function sets up the compose function, it also generates the audio for the voice-over.
    """
    voice_over_text = ui_params[0]
    # create the voice-over
    audio_f_path, vo_len_secs = create_voice_over_for_storyboard(voice_over_text, 1, DefaultRender.seconds)
    # recalculate the storyboard params
    my_ren_p.num_frames_per_section = int((my_ren_p.fps * vo_len_secs) / my_ren_p.sections)
    my_ren_p.num_frames = my_ren_p.num_frames_per_section * my_ren_p.sections
    my_ren_p.seconds = vo_len_secs

    sb_prompts = [i.prompt for i in storyboard_params if i is not None]
    sb_seeds_list = [i.seed[0] for i in storyboard_params if i is not None]

    sb_prompt = StoryBoardPrompt(sb_prompts,
                                 my_ren_p.seconds,
                                 False
                                 )
    sb_seeds = StoryBoardSeed(sb_seeds_list,
                              [0, my_ren_p.seconds * .5,
                               my_ren_p.seconds]
                              )

    num_keyframes = int(my_ren_p.num_frames * .1)

    return audio_f_path, num_keyframes, sb_prompt, sb_seeds, ui_params


def do_testing(my_ren_p, ui_params, test=-10):
    voice_over_text = "one two three four five six seven eight nine ten"
    voice_over_text = long_story_test_prompt

    # voice_over_text = small_quick_test_prompt_base
    # negative_prompt = small_quick_test_neg_prompt

    # sb_prompts = [
    #    small_quick_test_prompt1,
    #    small_quick_test_prompt2,
    #    small_quick_test_prompt3,
    # ]

    voice_over_text = short_story_test_prompt
    negative_prompt = short_story_test_neg_prompt

    sb_prompts = [
        short_story_sb_prompt1,
        short_story_sb_prompt2,
        short_story_sb_prompt3,
    ]

    # create the voice-over
    audio_f_path, vo_len_secs = create_voice_over_for_storyboard(voice_over_text, 1, DefaultRender.seconds)

    # sb_prompts = [
    #    "dog:1.0 cat:0.0",
    #    "dog:0.5 cat:0.5",
    #    "dog:0.0 cat:1.0",
    # ]

    # recalculate the storyboard params
    # vo_len_secs = 10.0
    my_ren_p.fps = 8
    my_ren_p.num_frames_per_section = int((my_ren_p.fps * vo_len_secs) / my_ren_p.sections)
    my_ren_p.num_frames = my_ren_p.num_frames_per_section * my_ren_p.sections
    my_ren_p.seconds = vo_len_secs

    sb_seeds_list = [1, 2, 3]
    ui_params = list(ui_params)
    ui_params[1] = negative_prompt
    ui_params[2] = 6  # steps

    sb_prompt = StoryBoardPrompt(sb_prompts, my_ren_p.seconds, False)
    sb_seeds = StoryBoardSeed(sb_seeds_list,
                              [0, my_ren_p.seconds * .5,
                               my_ren_p.seconds]
                              )
    num_keyframes = int(my_ren_p.num_frames * .1)

    return audio_f_path, num_keyframes, sb_prompt, sb_seeds, ui_params


def compose_storyboard_render_dev(my_ren_p, storyboard_params, ui_params, sb_rend_func, test=False,
                                  early_stop=-1):
    """
    this function composes the other rendering function to render a storyboard
    :param my_ren_p: the render parameters for the storyboard
    :param storyboard_params: the storyboard parameters
    :param ui_params: the ui parameters
    :param sb_rend_func: the render function to use for rendering the SBMultiSampleArgs
    :param test: if true, then the function will perform a quick test render
    :param early_stop: if not -1, then the function will stop after this many seconds
    """
    start_time = time.time()
    my_ren_p.width = ui_params[4]
    my_ren_p.height = ui_params[5]
    rend_func = None

    if test:
        audio_f_path, num_keyframes, sb_prompt, sb_seeds, ui_params = \
            do_testing(my_ren_p, ui_params, sb_rend_func)

    elif not test:
        audio_f_path, \
        num_keyframes, \
        sb_prompt, \
        sb_seeds, \
        ui_params = \
            do_compose_setup(my_ren_p,
                             ui_params,
                             storyboard_params)

    num_frames = my_ren_p.num_frames
    minimum_via_diff = .012  # this is a magic number, found by trial and error at 24 fps

    all_imgs_by_seconds = sync_renderer(minimum_via_diff,
                                        num_frames,
                                        num_keyframes,
                                        sb_rend_func,
                                        sb_prompt,
                                        sb_seeds,
                                        my_ren_p.seconds,
                                        ui_params,
                                        my_ren_p)

    target_mp4_f_path = process_aync_results(all_imgs_by_seconds, audio_f_path, my_ren_p, start_time)
    return target_mp4_f_path


def process_aync_results(all_imgs_by_seconds, audio_f_path, my_ren_p, start_time):
    # all_imgs_by_seconds:((int,{float:Image}),...)
    # sort the pairs by the int
    tmp = list(all_imgs_by_seconds)
    tmp.sort(key=lambda x: x[0])
    new = []
    for i in tmp:
        new.append(i[1])
    all_imgs_by_seconds = new
    if isinstance(all_imgs_by_seconds, tuple):
        tmp = all_imgs_by_seconds[0]
        tmp.update(all_imgs_by_seconds[1])
        all_imgs_by_seconds = tmp
    img_out = []
    for slice in all_imgs_by_seconds:
        for img, v in slice.items():
            if "frame_type" not in v.info:
                v.info["frame_type"] = "unknown"
            logger.info(msg=f'time: {img}, frame type: {v.info["frame_type"]}')
            img_out.append(v)
    # all_imgs_by_seconds = dict(sorted(all_imgs_by_seconds.items()))
    # for k, v in all_imgs_by_seconds.items():
    #    if "frame_type" not in v.info:
    #        v.info["frame_type"] = "unknown"
    #    logger.info(msg=f'time: {k}, frame type: {v.info["frame_type"]}')
    logger.info(msg=f'total of {len(img_out)} frames')
    images_to_save = img_out  # [i for i in all_imgs_by_seconds.values()]
    target_mp4_f_path = compose_file_handling(audio_f_path, images_to_save, my_ren_p.fps, my_ren_p.width,
                                              my_ren_p.height)
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time}")
    return target_mp4_f_path


if __name__ == "__main__":
    """this is the cli entry point, which also allows for testing."""
    choice = input("mode:\n\t(q)uick test\n\t(s)hort test\n\t(5)0\n\t(r)eal usage\n\t(e)xit\n")
    if choice == "q":  # quick test
        from modules.storysquad_storyboard.testing import get_test_storyboard

        test_sb = get_test_storyboard()
        storyboard_params = test_sb
        test_prompt = False

        test_ui_params = ["doggy dog dogg", "nude", 7, 3, 512, 512, 6, 7, 8, 9, 10, 11, 12,
                          7.5]  # only used if test is False

    elif choice == "s":  # short test
        raise NotImplementedError

    elif choice == "5":  # long test
        import modules.storysquad_storyboard.constants as sb_constants
        from modules.storysquad_storyboard.testing import get_test_storyboard
        print("using quick 50")
        test_prompt = False
        storyboard_params = [sb_constants.fifty_word_story1,
                             sb_constants.fifty_word_story2,
                             sb_constants.fifty_word_story3]

        speach = " ".join([i for i in storyboard_params[0] if not i.isdigit()])
        sbp_out = get_test_storyboard(storyboard_params)

        storyboard_params = sbp_out

        # remove all numbers from the prompt



        test_ui_params = [speach, "", 7, 3, 512, 512, 6, 7, 8, 9, 10, 11, 12, 7.5]  # only used if test is False


    server_choice = input("server:\n\t(m)ock\n\t(l)ocal\n\t(r)eal\n")

    if server_choice == "m":  # use the mock server controller to test
        import modules.storysquad_storyboard.env as sb_env

        sb_env.STORYBOARD_PRODUCT = "soloui"
        sb_env.STORYBOARD_SERVER_CONTROLLER_URL = "http://127.0.0.1:5000"
        sb_env.STORYBOARD_USE_AWS = True
        sb_env.STORYBOARD_API_ROLE = "ui_only"

    elif server_choice == "r":  # use the real server controllor to test
        import modules.storysquad_storyboard.env as sb_env

        print(
            f"using real server controller, press control-c now to interrupt. {sb_env.STORYBOARD_SERVER_CONTROLLER_URL}")
        time.sleep(5)
        sb_env.STORYBOARD_PRODUCT = "soloui"
        # sb_env.STORYBOARD_SERVER_CONTROLLER_URL = "http://
        sb_env.STORYBOARD_USE_AWS = True
        sb_env.STORYBOARD_API_ROLE = "ui_only"

    elif server_choice == "l":  # use the local server controllor to test
        import modules.storysquad_storyboard.env as sb_env

        sb_env.STORYBOARD_PRODUCT = "soloui"
        sb_env.STORYBOARD_USE_AWS = False
        sb_env.STORYBOARD_API_ROLE = "ui_only"
        sb_env.STORYBOARD_RENDER_SERVER_URLS = ["http://127.0.0.1:7861/sdapi/v1/txt2img"]

    from modules.storysquad_storyboard.sb_sd_render import storyboard_call_endpoint, shutdown_all_render_servers

    render_params = DefaultRender()  # only used if test is False
    render_params.fps = 30  # only used if test is False
    render_params.batch_size = 30  # only used if test is False

    compose_storyboard_render_dev(render_params,
                                  storyboard_params,
                                  test_ui_params,
                                  storyboard_call_endpoint,
                                  test=test_prompt
                                  )
    asyncio.run(shutdown_all_render_servers())
