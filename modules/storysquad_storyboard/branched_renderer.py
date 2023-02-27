import random
from typing import Callable, Any

import numpy as np

from modules.storysquad_storyboard.sb_rendering import DefaultRender, create_voice_over_for_storyboard, \
    SBMultiSampleArgs, save_tmp_images_for_debug, get_frame_deltas, MAX_BATCH_SIZE, get_linear_interpolation, \
    get_img_diff, compose_file_handling

from modules.storysquad_storyboard.storyboard import StoryBoardPrompt, StoryBoardSeed, StoryBoardData, SBIHyperParams


def compose_storyboard_render_dev(my_ren_p, storyboard_params, ui_params, render_func, test=False,
                                  early_stop=-1):
    """
    this function composes the other rendering function to render a storyboard
    :param my_ren_p: the render parameters for the storyboard
    :param storyboard_params: the storyboard parameters
    :param ui_params: the ui parameters
    :param render_func: the render function to use for rendering the SBMultiSampleArgs
    :param test: if true, then the function will perform a quick test render
    :param early_stop: if not -1, then the function will stop after this many seconds
    >>> test_ui_params = ["test","nude",7,3,4,5,6,7,8,9,10,11,12,7.5]
    >>> compose_storyboard_render_dev(DefaultRender(),None,test_ui_params ,lambda x: random.random() ,test=True)
    """

    if test:
        voice_over_text = "one two three four five six seven eight nine ten"
    else:
        voice_over_text = ui_params[0]

    # create the voice over
    audio_f_path, vo_len_secs = create_voice_over_for_storyboard(voice_over_text, 1, DefaultRender.seconds)
    # recalculate the storyboard params
    my_ren_p.num_frames_per_section = int((my_ren_p.fps * vo_len_secs) / my_ren_p.sections)
    my_ren_p.num_frames = my_ren_p.num_frames_per_section * my_ren_p.sections
    my_ren_p.seconds = vo_len_secs

    if test:
        sb_prompts = [
            "dog:1.0 cat:0.0",
            "dog:0.5 cat:0.5",
            "dog:0.0 cat:1.0",
        ]
        vo_len_secs = 60
        my_ren_p.num_frames_per_section = int((my_ren_p.fps * vo_len_secs) / my_ren_p.sections)
        my_ren_p.num_frames = my_ren_p.num_frames_per_section * my_ren_p.sections
        my_ren_p.seconds = vo_len_secs
        sb_seeds_list = [1, 2, 3]

    else:
        sb_prompts = [i.prompt for i in storyboard_params]
        sb_seeds_list = [i.seed for i in storyboard_params]

    sb_prompt = StoryBoardPrompt(sb_prompts, my_ren_p.seconds, False)
    sb_seeds = StoryBoardSeed(sb_seeds_list,
                              [my_ren_p.seconds * .5,
                               my_ren_p.seconds]
                              )
    sb_data = StoryBoardData(
        storyboard_seed=sb_seeds,
        storyboard_prompt=sb_prompt
    )

    # TODO: need to find seeds/subseeds/weights for each prompt

    ez_p_func: Callable[[Any], SBIHyperParams] = lambda ti: \
        SBIHyperParams(prompt=sb_prompt[ti],
                       negative_prompt=ui_params[1],
                       steps=ui_params[2],
                       seed=sb_seeds.get_prime_seeds_at_times(ti),
                       subseed=sb_seeds.get_subseeds_at_times(ti),
                       subseed_strength=sb_seeds.get_subseed_strength_at_times(ti),
                       cfg_scale=ui_params[13]
                       )

    ez_r_func: Callable[[Any], Any] = lambda x: render_func(SBMultiSampleArgs(hyper=ez_p_func(x), render=my_ren_p))

    num_keyframes = int(my_ren_p.num_frames * .1)

    all_imgs_by_seconds = {
        np.float64(0): ez_r_func(0.0).all_images[-1],
        np.float64(my_ren_p.seconds): ez_r_func(my_ren_p.seconds).all_images[-1],
    }
    all_imgs_by_seconds[0.0].info = {"frame_type": "g"}
    all_imgs_by_seconds[my_ren_p.seconds].info = {"frame_type": "g"}

    # imgs_pairs_by_diff = {
    #    get_img_diff(imgs_by_seconds[0], imgs_by_seconds[my_ren_p.seconds]): (
    #        list(imgs_by_seconds.keys())[0], list(imgs_by_seconds.keys())[1])
    # }

    done_pairs = []
    loop_count = -1
    minimum_via_diff = .012
    while True:
        loop_count += 1
        all_imgs_by_seconds = dict(sorted(all_imgs_by_seconds.items()))

        # save every 10th frame for debug
        if loop_count % 10 == 0:
            save_tmp_images_for_debug(all_imgs_by_seconds, loop_count)

        # non_interp_imgs_by_seconds = \
        ##    {k: v for k, v in all_imgs_by_seconds.items() if
        #     "frame_type" not in v.info or v.info["frame_type"] != "i"}

        imgs_by_seconds_keys_list = list(all_imgs_by_seconds.keys())
        # find the largest difference in means between two records
        frame_deltas = get_frame_deltas(list(all_imgs_by_seconds.values()))

        candidate_img_pairs_by_diff = []  # [delta, seconds_idx_1, seconds_idx_2]
        for delta_idx in range(len(frame_deltas)):
            time_pair = (imgs_by_seconds_keys_list[delta_idx], imgs_by_seconds_keys_list[delta_idx + 1])
            time_pair_img_infos = [all_imgs_by_seconds[time_pair[0]].info,
                                   all_imgs_by_seconds[time_pair[1]].info]

            if time_pair not in done_pairs:
                candidate_img_pairs_by_diff.append(
                    (frame_deltas[delta_idx],
                     (imgs_by_seconds_keys_list[delta_idx],
                      imgs_by_seconds_keys_list[delta_idx + 1])
                     )
                )
            else:
                print(f"skipping {time_pair} because it was already done")
        # update the list to be scores based on the delta and if the frame is an i type
        new = []
        for k, v in candidate_img_pairs_by_diff:
            if "nan" in all_imgs_by_seconds[v[0]].info["frame_type"]:
                new.append((k * 0, v))
            else:
                new.append((k, v))
        candidate_img_pairs_by_diff = new
        # sort so that the largest difference is first
        imgs_pairs_by_diff_sorted = sorted(candidate_img_pairs_by_diff, key=lambda x: x[0], reverse=True)

        maxdiff = imgs_pairs_by_diff_sorted[0][0]
        print(f"worst pair diff: {maxdiff},mean: {np.mean(frame_deltas)}")
        # if the largest difference is less than some value then we are done
        # if we have rendered enough images, then stop

        if maxdiff < minimum_via_diff:
            if maxdiff <  (minimum_via_diff * .1):
                if len(all_imgs_by_seconds) >= int(my_ren_p.num_frames * .8):
                    print(f'done because of low maxdiff')
                    break

            if len(all_imgs_by_seconds) >= int(my_ren_p.num_frames * 1.2):
                print(f'done because of too many frames")')
                break

        # get a batch of the worst images
        worst_pairs_batch = imgs_pairs_by_diff_sorted[:MAX_BATCH_SIZE]
        # remove all pairs from the batch with a diff less than the minimum via diff
        worst_pairs_batch = [v for v in worst_pairs_batch if v[0] >= minimum_via_diff]

        # remove any piars that have an i type frame
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
            if len(all_imgs_by_seconds) >= int(my_ren_p.num_frames * .8):
                print(f"done due to nothing to do")
                break
            else:
                print(f"lowing target maxdiff because of nothing to do")
                minimum_via_diff = minimum_via_diff * .9
                print(f"new minimum_via_diff: {minimum_via_diff} old: {minimum_via_diff * 1.1}")
                continue

        # update done_pairs
        pairs_to_gen = [v[1] for v in worst_pairs_gen_batch]
        done_pairs.append(pairs_to_gen)

        # get the target times for each pair
        target_times = [np.mean(v[1:]) for v in worst_pairs_gen_batch]

        # render the batch
        if len(target_times) > 0:
            imgs = ez_r_func(target_times).all_images
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
            segmnt_per_of_time = time_of_segment / sb_prompt.total_seconds

            diff_a = get_img_diff(imga, g_img)
            diff_b = get_img_diff(g_img, imgb)
            diff_mean = (diff_a + diff_b) / 2

            # this is the difference between the two images
            # TODO: this maybe can be safely retrieved from frame_deltas
            diff_orig = get_img_diff(imga, imgb)


            if diff_a <=0.0002 or diff_b <= 0.0002:
                print(f'frame {t_of_f} from pair {src_pair} must be i frame because a diff is 0')
                frame_type = "i"
            elif len(all_imgs_by_seconds) < num_keyframes:
                frame_type = "keyframe"
            # elif time_of_segment < (1 / my_ren_p.fps):  # if the segment is less than a frame
            #    frame_type = "i"
            #    print(f'frame {t_of_f} from pair {src_pair} must be i frame because segment is less than a frame')
            elif diff_a <= diff_orig and diff_b < diff_orig:
                frame_type = "g"
                print(f'frame {t_of_f} from pair {src_pair} must be g frame because a diff is less than orig')
            else:
                frame_type = "i"

            if frame_type == "keyframe" or frame_type == "g":
                print(f'adding -{frame_type}- frame @ time: {t_of_f} for pair: {src_pair}')
                g_img.info["frame_type"] = "g"
                all_imgs_by_seconds[t_of_f] = g_img
            elif frame_type == "s":
                print(f'skip adding -{frame_type}- frame @ time: {t_of_f} for pair: {src_pair}')
            elif frame_type == "i":
                # report what we did not add
                # print( f'not adding time: {t} for pair: {p}')
                # add a linear interpolation between the two images
                # print(f'adding -{frame_type}- (interpolation) frame between {src_pair[0]} and {src_pair[1]}')
                i_frames_needed = round(diff_orig / 3) + 1
                i_frames_needed = min(i_frames_needed, 1)

                print(
                    f'adding {i_frames_needed} -{frame_type}- (interpolation) frames between {src_pair[0]} and {src_pair[1]}')
                # imga.info["frame_type"] += "i"
                # imgb.info["frame_type"] = "i" + imgb.info["frame_type"]
                q = i_frames_needed
                p_s = np.interp(fp=[0, 1],
                                xp=[0, 1],
                                x=[float(i) for i in np.arange(1 / (1 + q),
                                                               0.9999999999,
                                                               1 / (1 + q))
                                   ])
                # for p in p_s:
                #    done_pairs.append((src_pair[0],
                #                       src_pair[0] + p * time_of_segment))

                for p in p_s:
                    k = p * time_of_segment + src_pair[0]
                    v = get_linear_interpolation(imga, imgb, p)
                    v.info["frame_type"] = "i"
                    all_imgs_by_seconds[k] = v

                ## remove the pair from the done_pairs
                # done_pairs = [v for v in done_pairs if v != src_pair]

    for k, v in all_imgs_by_seconds.items():
        print(k)
    print(f'total of {len(all_imgs_by_seconds)} frames')

    images_to_save = [i for i in all_imgs_by_seconds.values()]
    for k, v in all_imgs_by_seconds.items():
        v.save(f'tmpE_{k}.png')

    target_mp4_f_path = compose_file_handling(audio_f_path, images_to_save)
    return target_mp4_f_path
