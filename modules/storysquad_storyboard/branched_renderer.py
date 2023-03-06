import random
import logging
import time
from PIL import Image
from typing import Callable, Any
from modules.storysquad_storyboard.sb_rendering import SBImageResults
import numpy as np
from modules.storysquad_storyboard.constants import *

logging.basicConfig(filename="branched_renderer.log",level=1)
logger:logging.Logger = logging.getLogger("BranchedRenderer")
logger.setLevel(logging.DEBUG)
logger.info("entering branched renderer")

from modules.storysquad_storyboard.sb_rendering import DefaultRender, create_voice_over_for_storyboard, \
    SBMultiSampleArgs, save_tmp_images_for_debug, get_frame_deltas, MAX_BATCH_SIZE, get_linear_interpolation, \
    get_img_diff, compose_file_handling

from modules.storysquad_storyboard.storyboard import StoryBoardPrompt, StoryBoardSeed, StoryBoardData, SBIHyperParams


def do_testing(my_ren_p, storyboard_params, ui_params, render_func, test=-10):
    voice_over_text = "one two three four five six seven eight nine ten"
    voice_over_text = long_story_test_prompt
    voice_over_text = short_story_test_prompt
    negative_prompt = short_story_test_neg_prompt

    # create the voice-over
    audio_f_path, vo_len_secs = create_voice_over_for_storyboard(voice_over_text, 1, DefaultRender.seconds)

    sb_prompts = [
        "dog:1.0 cat:0.0",
        "dog:0.5 cat:0.5",
        "dog:0.0 cat:1.0",
    ]

    sb_prompts = [
        short_story_sb_prompt1,
        short_story_sb_prompt2,
        short_story_sb_prompt3,
    ]

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
                              [my_ren_p.seconds * .5,
                               my_ren_p.seconds]
                              )
    rend_func = get_rend_func(sb_prompt, sb_seeds, ui_params, my_ren_p, render_func)
    num_keyframes = int(my_ren_p.num_frames * .1)

    return audio_f_path, num_keyframes, rend_func, sb_prompt


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
    >>> test_ui_params = ["test","nude",7,3,4,5,6,7,8,9,10,11,12,7.5]
    >>> compose_storyboard_render_dev(DefaultRender(),None,test_ui_params ,lambda x: random.random() ,test=True)
    """
    start_time = time.time()
    if test == True:
        audio_f_path, num_keyframes, rend_func, sb_prompt = \
            do_testing(my_ren_p, storyboard_params, ui_params, sb_rend_func)

    elif test == False:
        audio_f_path, num_keyframes, rend_func, sb_prompt = do_compose_setup(my_ren_p, storyboard_params, test,
                                                                             ui_params, sb_rend_func)
    rend_func: Callable[[float], SBImageResults] = rend_func

    num_frames = my_ren_p.num_frames
    minimum_via_diff = .012  # this is a magic number, found by trial and error at 24 fps

    all_imgs_by_seconds = renderer(minimum_via_diff,
                                   num_frames,
                                   num_keyframes,
                                   rend_func,
                                   sb_prompt,
                                   my_ren_p.seconds)

    for k, v in all_imgs_by_seconds.items():
        logger.info(msg=k)
    logger.info(msg=f'total of {len(all_imgs_by_seconds)} frames')

    images_to_save = [i for i in all_imgs_by_seconds.values()]
    images_to_save = [i for i in all_imgs_by_seconds.values()]

    target_mp4_f_path = compose_file_handling(audio_f_path, images_to_save, my_ren_p.fps)
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time}")
    return target_mp4_f_path


#def renderer(minimum_via_diff, num_frames, num_keyframes, rend_func, sb_prompt, seconds_length):
def renderer(minimum_via_diff: float, num_frames: int, num_keyframes: int, rend_func: callable, sb_prompt: str,
                 seconds_length: float) -> [Image]:
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
    # seed the image list with a batch of images
    all_imgs_by_seconds_times = [i for i in np.arange(0, seconds_length, seconds_length / MAX_BATCH_SIZE)]
    all_imgs_by_seconds_images = rend_func(all_imgs_by_seconds_times).all_images[-MAX_BATCH_SIZE:]
    all_imgs_by_seconds = dict(zip(all_imgs_by_seconds_times, all_imgs_by_seconds_images))
    # set the first batch of images to be keyframes
    for i in all_imgs_by_seconds:
        all_imgs_by_seconds[i].info["frame_type"] = "k"
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
                logger.info(f"skipping {time_pair} because it was already done")

        imgs_pairs_by_diff_sorted = sorted(candidate_img_pairs_by_diff, key=lambda x: x[0], reverse=True)

        maxdiff = imgs_pairs_by_diff_sorted[0][0]
        logger.info(f"worst pair diff: {maxdiff},mean: {np.mean(frame_deltas)}")

        # if the largest difference is less than some value then we are done
        # if we have rendered enough images, then stop
        if maxdiff < minimum_via_diff:
            if maxdiff < (minimum_via_diff * .1):
                if len(all_imgs_by_seconds) >= int(num_frames * .9):
                    logger.info(f'done because of low maxdiff')
                    break

        if len(all_imgs_by_seconds) >= int(num_frames * 1.1):
            logger.info(f'done because of too many frames")')
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
                logger.info(f"done due to nothing to do")
                break
            else:
                # this is not necessarily necessary...up to Product team.
                # this means that we are not done, but we don't have enough frames to be done,
                # so we need to lower the minimum via diff to get more frames
                logger.info(f"lowing target maxdiff because of nothing to do")
                minimum_via_diff = minimum_via_diff * .9
                logger.info(f"new minimum_via_diff: {minimum_via_diff} old: {minimum_via_diff * 1.1}")
                continue

        # update done_pairs
        pairs_to_gen = [v[1] for v in worst_pairs_gen_batch]
        done_pairs.append(pairs_to_gen)

        # get the target times for each pair
        target_times = [np.mean(v[1:]) for v in worst_pairs_gen_batch]

        # render the batch
        if len(target_times) > 0:
            imgs = rend_func(target_times).all_images
            imgs = imgs[1:] or [imgs[0]]
            pairs_done_this_iter = pairs_to_gen
            del pairs_to_gen

        # interpolate the i type frames
        for pair in worst_pairs_int_batch:
            # get the two images
            imga = all_imgs_by_seconds[pair[1][0]]
            imgb = all_imgs_by_seconds[pair[1][1]]
            # get the time of the frame
            time_of_segment = pair[1][1] - pair[1][0]
            segmnt_per_of_time = time_of_segment / sb_prompt.total_seconds
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
            # segmnt_per_of_time = time_of_segment / sb_prompt.total_seconds

            diff_a = get_img_diff(imga, g_img)
            diff_b = get_img_diff(g_img, imgb)
            # diff_mean = (diff_a + diff_b) / 2

            # this is the difference between the two images
            # TODO: this maybe can be safely retrieved from frame_deltas
            diff_orig = get_img_diff(imga, imgb)

            if diff_a <= 0.0002 or diff_b <= 0.0002:
                logger.info(f'frame {t_of_f} from pair {src_pair} must be i frame because a diff is 0')
                frame_type = "i"
            elif len(all_imgs_by_seconds) < num_keyframes:
                frame_type = "keyframe"
            # elif time_of_segment < (1 / my_ren_p.fps):  # if the segment is less than a frame
            #    frame_type = "i"
            #    logger.info(f'frame {t_of_f} from pair {src_pair} must be i frame because segment is less than a frame')
            elif diff_a <= diff_orig and diff_b < diff_orig:  # if the difference is less than the original difference
                frame_type = "g"
                logger.info(f'frame {t_of_f} from pair {src_pair} must be g frame because a diff is less than orig')
            else:  # if the difference is greater than the original difference
                frame_type = "i"

            if frame_type == "keyframe" or frame_type == "g":
                logger.info(f'adding -{frame_type}- frame @ time: {t_of_f} for pair: {src_pair}')
                g_img.info["frame_type"] = "g"
                all_imgs_by_seconds[t_of_f] = g_img

            elif frame_type == "i":
                i_frames_needed = round(diff_orig / 3) + 1  # this is a constant I came to by trail and error
                i_frames_needed = min(i_frames_needed, 1)
                # the above actually needs to be changed because the difference score is now from 0 to 1 instead of 0 to 255

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
    return all_imgs_by_seconds


def do_compose_setup(my_ren_p, storyboard_params, ui_params, render_func):
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

    sb_prompts = [i.prompt for i in storyboard_params]
    sb_seeds_list = [i.seed for i in storyboard_params]

    sb_prompt = StoryBoardPrompt(sb_prompts, my_ren_p.seconds, False)
    sb_seeds = StoryBoardSeed(sb_seeds_list,
                              [my_ren_p.seconds * .5,
                               my_ren_p.seconds]
                              )
    rend_func = get_rend_func(sb_prompt, sb_seeds, ui_params, my_ren_p, render_func)

    num_keyframes = int(my_ren_p.num_frames * .1)
    return audio_f_path, num_keyframes, rend_func, sb_prompt


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
