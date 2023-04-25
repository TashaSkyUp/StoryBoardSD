from typing import List, Any, Optional
import copy
import random
from dataclasses import dataclass, field
from typing import List

import PIL
import gtts
import numpy
from PIL import Image

from modules.storysquad_storyboard.env import *
import modules.storysquad_storyboard.env as sb_env
from modules.storysquad_storyboard.storyboard import SBIHyperParams, get_prompt_words_and_weights_list, \
    get_frame_seed_data
from modules.storysquad_storyboard.constants import *

GTTS_SAMPLE_RATE = 24000.0
MAX_BATCH_SIZE = sb_env.STORYBOARD_MAX_BATCH_SIZE
NUM_SB_IMAGES = 30


@dataclass
class DefaultRender:
    fps: int = 24
    minutes: int = 2
    seconds: int = minutes * 60
    sections: int = 2
    num_frames: int = int(seconds * fps)
    min_frames_per_render: int = fps * 10
    num_frames_per_sctn: int = int(num_frames / sections)
    early_stop_seconds = int(60 * 30 * 16)  # 8 hours
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
    batch_count: int = 1
    batch_size: int = MAX_BATCH_SIZE
    sampler_index: int = 8  # 9
    sampler_name: str = "DPM2 Karras"


@dataclass
class SBIRenderParams:
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
    batch_count: int = 1
    batch_size: int = MAX_BATCH_SIZE
    sampler_index: int = 8  # 9
    sampler_name: str = "DPM2 Karras"


def quick_timer(func, *args, **kwargs):
    """
    this is a quick timer to time functions
    outputs the time and the result of the function

    """
    import time
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end - start, result


import numpy as np


def get_img_diff(image1: PIL.Image, image2: PIL.Image, K1=0.01, K2=0.03, L=255):
    """
    Calculate the structural similarity index between two color images, accounting for the human visual system.

    Parameters:
    image1 (numpy.ndarray): An RGB image as a numpy array.
    image2 (numpy.ndarray): An RGB image as a numpy array, with the same shape as image1.
    K1 (float): A stability constant for SSIM calculation. Default is 0.01. K1 controls the
                degree of luminance comparison in the SSIM calculation. Higher values of K1
                give more weight to differences in luminance between the two images.
    K2 (float): A stability constant for SSIM calculation. Default is 0.03. K2 controls the
                degree of contrast comparison in the SSIM calculation. Higher values of K2
                give more weight to differences in contrast between the two images.
    L (float): The dynamic range of the pixel values. Default is 255.

    Returns:
    float: A value between 0 and 1 representing the difference between the two images. A value of 0 indicates that
           the images are identical, while a value of 1 indicates that they are completely dissimilar.

    Examples:
    >>> import numpy as np
    >>> img1 = np.zeros((10, 10, 3))*255
    >>> img2 = np.ones((10, 10, 3))*255
    >>> get_img_diff(img1, img2)
    0.999899990025949
    >>> img2 = np.zeros((10, 10, 3))*255
    >>> get_img_diff(img1, img2)
    0.0
    """
    # Convert the images to numpy
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same shape.")

    # Convert images to grayscale using weighted average of color channels
    gray1 = 0.2989 * image1[..., 0] + 0.5870 * image1[..., 1] + 0.1140 * image1[..., 2]
    gray2 = 0.2989 * image2[..., 0] + 0.5870 * image2[..., 1] + 0.1140 * image2[..., 2]

    # Calculate mean and variance of input images
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    var1 = np.var(gray1)
    var2 = np.var(gray2)
    covar = np.cov(gray1.ravel(), gray2.ravel())[0][1]

    # Set constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator

    # Return difference score
    return 1 - ssim


def join_video_audio(video_file, audio_file):
    from moviepy.editor import VideoFileClip, AudioFileClip
    import random
    video = VideoFileClip(video_file)
    audio = AudioFileClip(audio_file)
    video.audio = audio
    # create a random name for the output file
    rnd_file = f"tmp_{random.randint(0, 1000000)}.mp4"
    video.write_videofile(rnd_file)
    return rnd_file


def remove_silence(data, data_sample_rate, desired_length_secs):
    """removes up to 20% of the audio that is closer to being considered silence.
    this is done by calculating how to long the audio would be if values below n percentile were removed
    using a smoothed 1000 sample representation of the audio
    """
    import numpy as np
    aud_len = len(data) / data_sample_rate
    if desired_length_secs / aud_len < .8:
        desired_length_secs = aud_len * .8

    data = data[0, :]
    conv = np.array(np.sin(np.arange(0, np.pi, np.pi / 1000)))[1:]
    conved = np.convolve(data, np.array(conv) / sum(conv), 'valid')

    sample_in_per_sample_out = int(len(conved) / 1000)
    out = np.zeros(0)
    conved = conved[::sample_in_per_sample_out]
    conved = conved / np.max(np.abs(conved))

    conv_sec_len = aud_len / 1000
    conved = np.abs(conved)
    tester = [conved[conved < np.percentile(conved, i / 10)].shape[0] * conv_sec_len for i in range(1, 1000)]
    tester = np.array(tester)
    percentile_to_use = np.abs(tester - desired_length_secs).argmin()
    percentile_value = np.percentile(conved, percentile_to_use / 10)

    for i in range(1000):
        if conved[i] < percentile_value:
            out = np.append(out, data[i * sample_in_per_sample_out:(i + 1) * sample_in_per_sample_out])
        else:
            pass

    out = np.stack([out, out])
    return out


def create_voice_over_for_storyboard(text_to_read, speech_speed, vo_length_sec):
    """
    >>> while True:
    ...  import os
    ...  # print the current directory
    ...  print(os.getcwd())
    ...  create_voice_over_for_storyboard(long_story_test_prompt, None, 10)
    ...  break
    """
    # time everything

    print("on_preview_audio")
    save_sample_rate = 44100
    # get the audio samples to aud_out while timing it

    t, (aud_out, aud_length_secs) = quick_timer(get_samples_from_gtts, text_to_read)
    print(f"get_samples_from_gtts: latency {t}")

    t, (aud_out_slow, aud_length_secs_slow) = quick_timer(get_samples_from_gtts, text_to_read, slow=True)
    reg_dist = (vo_length_sec - aud_length_secs) ** 2
    slow_dist = (vo_length_sec - aud_length_secs_slow) ** 2
    if slow_dist < reg_dist:
        aud_out = aud_out_slow
        audio_length_secs = aud_length_secs_slow

    t, (data, data_sample_rate) = quick_timer(robot_voice_effect, aud_out, iterations=0)
    print(f'robot_voice_effect latency: {t}')

    audio_length_secs = max(*data.shape) / data_sample_rate
    if audio_length_secs > vo_length_sec:
        # remove silence of longer than .5 seconds
        t, data = quick_timer(remove_silence, data, data_sample_rate, vo_length_sec)
        print(f'remove_silence latency: {t}')

    audio_length_secs = max(*data.shape) / data_sample_rate

    rnd_file = write_mp3(data, data_sample_rate, save_sample_rate)
    # rnd_file = os.path.join(STORYBOARD_TMP_PATH, rnd_file)
    return rnd_file, audio_length_secs


def write_mp3(effected, data_sample_rate, save_sample_rate):
    import numpy as np
    import random
    # import AudioArrayClip from moviepy
    from moviepy.audio.AudioClip import AudioArrayClip

    rnd_file = f"tmp_{random.randint(0, 1000000)}.wav"
    rnd_file = os.path.join(STORYBOARD_TMP_PATH, rnd_file)
    # write the audio to
    afc = AudioArrayClip(np.moveaxis(effected, 0, -1), data_sample_rate)
    afc.write_audiofile(rnd_file, fps=save_sample_rate)
    return rnd_file


def robot_voice_effect(aud_out, iterations=4):
    """
    >>> voice = get_samples_from_gtts("one two three four five six seven eight nine ten ")[0]
    >>> robot_voice_effect(voice)
    """
    import numpy as np

    rate = 44100
    print(aud_out.shape)
    effected = np.copy(aud_out)
    divisor = iterations

    effected = effected[:, ::divisor + 1]
    offset = int(rate / 125)
    for i in range(iterations):
        effected = effected[:, :effected.shape[1] - offset] + effected[:, offset:]

    # normalize
    effected = effected / np.max(np.abs(effected))
    print(effected.shape)
    rate = int(rate / (divisor + 1))

    return effected, rate


def rfind_list(text, to_find: list):
    """
    returns the index of the last item in to_find that is found in text plus its length,
     or the length of the test if none are found, also resorts to finding " " or "," if none are found


    >>> test = "one two three four five six seven eight nine ten "
    >>>
    >>> print(f"max_idx {len(test)-1} found at:{rfind_list(test, ['one', 'e','z'])}")
    max_idx 48 found at:47
    >>> print(f"max_idx {len(test)-1} found at:{rfind_list(test, ['one', 'two','agafd'])}")
    max_idx 48 found at:7
    >>> print(f"max_idx {len(test)-1} found at:{rfind_list(test, ['one', 'agafd'])}")
    max_idx 48 found at:3
    >>> print(f"max_idx {len(test)-1} found at:{rfind_list(test, ['apple', 'sauce'])}")
    max_idx 48 found at:-1
    """
    idx = -1
    finds = [text.rfind(i) + len(i) for i in to_find if text.rfind(i) != -1]
    finds.sort()
    if len(finds) == 0:
        low_priority = [',']
        finds = [text.rfind(i) + len(i) for i in low_priority if text.rfind(i) != -1]
        finds.sort()
        if len(finds) == 0:
            low_priority = [' ']
            finds = [text.rfind(i) + len(i) for i in low_priority if text.rfind(i) != -1]
            finds.sort()
            if len(finds) == 0:
                return len(text)
            else:
                return finds[-1]

        else:
            ret = finds[-1]
            return ret

    return finds[-1]


def get_samples_from_gtts(text_to_read, slow=False) -> (numpy.ndarray, float):
    """
    >>> get_samples_from_gtts(long_story_test_prompt)
    ... result_file=write_mp3(out, 44100,44100)
    """
    # TODO: this needs paraellization
    punctuations = '''!()[]{};:\\<>./?@#$%^&*_~\n'''
    punc = list(punctuations)
    punc.append("   ")

    # slice up the text_to_read into 100 character chunks or smaller than shrink them to the nearest punctuation
    from gtts import gTTS

    sections = gtts.gTTS(text_to_read, lang="en", slow=slow)._tokenize(text_to_read)

    from pedalboard.io import AudioFile
    import moviepy.editor as mpy
    import os

    aud_length_secs = 0
    out = []
    for section in sections:
        if section != " " and section != "":
            tmp_mp3_full_path = os.path.join(STORYBOARD_TMP_PATH, "tmp.mp3")
            tmp_wav_full_path = os.path.join(STORYBOARD_TMP_PATH, "tmp.wav")
            audio = gTTS(
                text=section,
                lang="en",
                slow=slow,
            )

            audio.save(tmp_mp3_full_path)

            mpy.AudioFileClip(tmp_mp3_full_path).write_audiofile(tmp_wav_full_path)
            with AudioFile(tmp_wav_full_path, "r") as f:
                aud_out = f.read(f.frames)
            # get the length of the audio
            aud_length_secs += (aud_out.shape[1] / GTTS_SAMPLE_RATE)

            # delete the files

            try:
                os.remove(tmp_mp3_full_path)
            except PermissionError as e:
                print(f"PermissionError: {e}")
                print(f"trying to delete {tmp_mp3_full_path}")
                os.remove(tmp_mp3_full_path)

            os.remove(tmp_wav_full_path)
            out.append(aud_out)

    return np.concatenate(out, axis=1), aud_length_secs


def make_mp4(input_path, filepath, filename, width, height, keep, fps=30) -> str:
    # TODO: use moviepy instead
    import subprocess
    import os
    import glob
    import uuid
    # clean the input path
    input_path = os.path.abspath(input_path)

    image_input_path = os.path.join(input_path, "%05d.png")
    mp4_path = os.path.join(filepath, f"{str(filename)}.mp4")
    # check if the file exists, if it does, change mp4_path to include part of a uuid
    if os.path.exists(mp4_path):
        print(f"file exists, changing name to {mp4_path}")
        mp4_path = os.path.join(filepath, f"{str(filename)}_{str(uuid.uuid4()).split('-')[-1]}.mp4")
    mp4_path = mp4_path.split(".mp4")[0] + ".mp4"
    # make the mp4
    exec_full_path = STORYBOARD_FFMPEG_PATH
    cmd = [
        exec_full_path,
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', str(image_input_path),
        '-c:v', 'libx264',
        '-vf', 'scale=' + str(width) + ':' + str(height),
        '-pix_fmt', 'yuv420p',
        '-crf', '11',
        '-preset', 'veryfast',
        str(mp4_path)
    ]
    print(f'executing: {" ".join(cmd)}')
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    if keep == False:
        for ifile in glob.glob(input_path + "/*.png"):
            os.remove(ifile)
    return mp4_path


def make_mp4_from_images(image_list, filepath, filename, width, height, keep, fps=30,
                         filter_func=False) -> str:
    import os
    import gc
    import glob
    from PIL import Image
    """Make an mp4 from a list of images, using make_mp4"""

    if filter_func:
        image_list = filter_func(image_list)
        # convert numpy array to PIL image
        gc.collect()
        u_8_image_list = [np.multiply(image, 255).astype("uint8") for image in image_list]
        del image_list
        gc.collect()
        image_list = []
        image_list = [Image.fromarray(image) for image in u_8_image_list]

    # save the images to a temp folder
    temp_folder_path = os.path.join(filepath, "frames")
    # remove all files in the input path
    for file in glob.glob(os.path.join(temp_folder_path, "*")):
        os.remove(file)

    if not os.path.exists(temp_folder_path):
        try:
            os.mkdir(temp_folder_path)
        except OSError:
            # create the parent folder if it doesn't exist
            os.mkdir(os.path.dirname(temp_folder_path))
            os.mkdir(temp_folder_path)

    for i, img in enumerate(image_list):
        i_filename = os.path.join(temp_folder_path, f"{str(i).zfill(5)}.png")
        img.save(f"{i_filename}")

    # make the mp4

    return make_mp4(f"{temp_folder_path}", filepath, filename, width, height, keep, fps=fps)


def get_frame_values_for_prompt_word_weights(prompts, num_frames):  # list[sections[frames[word:weight tuples]]]
    """
    >>> while True:
    ...     sections = get_frame_values_for_prompt_word_weights([CallArgsAsData(prompt= "(dog:1) cat:0"),CallArgsAsData(prompt= "(dog:1) cat:1"),CallArgsAsData(prompt= "(dog:0) cat:1")],4)
    ...     for section in sections:
    ...         print(section)
    ...     break
    [[('dog', 1.0), ('cat', 0.0)], [('dog', 1.0), ('cat', 0.3333333333333333)], [('dog', 1.0), ('cat', 0.6666666666666666)], [('dog', 1.0), ('cat', 1.0)]]
    [[('dog', 1.0), ('cat', 1.0)], [('dog', 0.6666666666666667), ('cat', 1.0)], [('dog', 0.33333333333333337), ('cat', 1.0)], [('dog', 0.0), ('cat', 1.0)]]
    """
    # get the weights for each word of each prompt in the prompts list returns a list of lists of tuples
    words_and_weights_for_prompts = [get_prompt_words_and_weights_list(p) for p in prompts]

    # define the two distinct sections of the storyboard_call_multi
    # each section is composed of frames, each frame has different weights for each word (probably) which results
    # in a unique image for the animation
    sections = [
        [words_and_weights_for_prompts[0], words_and_weights_for_prompts[1]],
        # transition from lattice pt 1 to 2
        [words_and_weights_for_prompts[1], words_and_weights_for_prompts[2]],
        # transition from lattice pt 2 to 3
    ]

    # interpolate the weights linearly for each word in each section for each frame and return the sections

    sections_frames = []
    for section in sections:
        start: tuple(str, float) = section[0]
        end: tuple(str, float) = section[1]
        word_frame_weights = []
        for i in range(num_frames):
            frame_weights = []
            for word_idx, word_at_pos in enumerate(start):
                # format like: ('dog', 0.0)
                word_start_weight = start[word_idx][1]
                word_end_weight = end[word_idx][1]
                word_frame_weight = \
                    word_start_weight + (word_end_weight - word_start_weight) * (i / (num_frames - 1))
                frame_weights.append((word_at_pos[0], word_frame_weight))
            word_frame_weights.append(frame_weights)
        sections_frames.append(word_frame_weights)

    return sections_frames


def limit_per_pixel_change_slice_old(frames, max_change):
    """ limits the change in each pixel to be no more than max_change """
    import numpy as np
    from PIL import Image
    # change the frames to a numpy array if they are not and normalize the values to be between 0 and 1
    if isinstance(frames[0], Image.Image):
        frames = [np.array(f) / 255 for f in frames]
    for i in range(1, len(frames)):
        # get the difference between the current frame and the last frame
        diff = frames[i] - frames[i - 1]
        # get the absolute value of the difference
        abs_diff = np.abs(diff)
        # get the pixels that have changed more than max_change
        too_big = abs_diff > max_change
        # scale the pixels that have changed too much
        frames[i][too_big] = frames[i - 1][too_big] + np.sign(diff[too_big]) * max_change
    return frames


def limit_per_pixel_change_slice(frames, max_change):
    """
    Limits the change in each pixel to be no more than max_change.

    Parameters:
    frames: a list of numpy arrays or PIL Image objects representing the frames.
    max_change: a float representing the maximum change in each pixel allowed.

    Returns:
    A list of numpy arrays representing the modified frames.

    Examples:
    >>> frames = [
    ...     np.array([
    ...         [0, 0, 0],
    ...         [0, 0, 0]
    ...     ]),
    ...     np.array([
    ...         [0.1, 0.1, 0.1],
    ...         [0.1, 0.1, 0.1]
    ...     ])
    ... ]
    >>> max_change = 0.2
    >>> limit_per_pixel_change_slice(frames, max_change)
    [array([[0, 0, 0],
           [0, 0, 0]]), array([[0.1, 0.1, 0.1],
           [0.1, 0.1, 0.1]])]
    >>> frames = [
    ...     np.array([
    ...         [0, 0, 0],
    ...         [0, 0, 0]
    ...     ]),
    ...     np.array([
    ...         [0.5, 0.5, 0.5],
    ...         [0.5, 0.5, 0.5]
    ...     ])
    ... ]
    >>> max_change = 0.2
    >>> limit_per_pixel_change_slice(frames, max_change)
    [array([[0, 0, 0],
           [0, 0, 0]]), array([[0.2, 0.2, 0.2],
           [0.2, 0.2, 0.2]])]
    >>> frames = [
    ...     Image.fromarray((np.array([
    ...         [0, 0, 0],
    ...         [0, 0, 0]
    ...     ]) * 255).astype(np.uint8)),
    ...     Image.fromarray((np.array([
    ...         [0.5, 0.5, 0.5],
    ...         [0.5, 0.5, 0.5]
    ...     ]) * 255).astype(np.uint8))
    ... ]
    >>> max_change = 0.2
    >>> limit_per_pixel_change_slice(frames, max_change)
    [array([[0., 0., 0.],
           [0., 0., 0.]]), array([[0.2, 0.2, 0.2],
           [0.2, 0.2, 0.2]])]
    """
    import numpy as np
    from PIL import Image
    # change the frames to a numpy array if they are not and normalize the values to be between 0 and 1
    if isinstance(frames[0], Image.Image):
        frames = [np.array(f, dtype="float16") / 255 for f in frames]

    for i in range(1, len(frames)):
        # get the difference between the current frame and the last frame
        diff = frames[i] - frames[i - 1]
        # get the absolute value of the difference
        abs_diff = np.abs(diff)
        # get the pixels that have changed more than max_change
        too_big = abs_diff > max_change
        # scale the pixels that have changed too much
        frames[i][too_big] = frames[i - 1][too_big] + np.sign(diff[too_big]) * max_change
    return frames


def limit_per_pixel_change_slice_optical_flow(frames, max_change):
    """
    Limits the change in each pixel to be no more than max_change using optical flow.

    Parameters:
    frames: a list of numpy arrays or PIL Image objects representing the frames.
    max_change: a float representing the maximum change in each pixel allowed.

    Returns:
    A list of numpy arrays representing the modified frames.
    >>> test_limit_per_pixel_change_slice_optical_flow()
    """
    import cv2
    import numpy as np

    # change the frames to a numpy array if they are not and normalize the values to be between 0 and 1
    if isinstance(frames[0], Image.Image):
        np_frames = [np.array(f) for f in frames]
        gray_frames = [cv2.cvtColor(f.astype("uint8"), cv2.COLOR_RGB2GRAY) for f in np_frames]
        frames = [(f / 255).astype("float16") for f in np_frames]
    # create a blank image to use as the previous frame
    prev_frame = np.zeros_like(frames[0])
    out_frames = []
    for prev_idx, cur_idx in zip(range(len(frames) - 1), range(1, len(frames))):
        prev_frame = frames[prev_idx]  # float 0-1
        out_frames.append(prev_frame)
        cur_frame = frames[cur_idx]  # float 0-1
        # convert the frames to grayscale
        prev_gray = gray_frames[prev_idx]  # uint8 0-255
        curr_gray = gray_frames[cur_idx]  # uint8 0-255
        # compute the optical flow between the current frame and the previous frame
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 5, 64, 5, 7, 1.5,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)  # -1 to 1 ?
        # get the pixels that have changed more than max_change
        # get the pixels that have changed more than max_change
        too_big = np.power(flow, 2)
        too_big = np.sum(too_big, axis=2)
        too_big = np.power(too_big, .5)  # 0 to 1 representing distance moved, unknown units
        too_big_bool = too_big > max_change
        # add an extra dimension to the too_big array so that it has the same shape as prev_frame and flow
        # too_big = np.expand_dims(too_big, axis=-1)
        # scale the pixels that have changed too much
        over_the_limit = too_big - max_change
        over_the_limit[over_the_limit < 0] = 0
        over_the_limit_ratio = over_the_limit / too_big
        per_of_old = over_the_limit_ratio
        per_of_new = 1 - over_the_limit_ratio
        per_of_new = np.expand_dims(per_of_new, axis=-1)
        per_of_old = np.expand_dims(per_of_old, axis=-1)
        per_of_old[per_of_old < 0] = 0
        per_of_new[per_of_new < 0] = 0
        new_frame = (cur_frame * per_of_new) + (prev_frame * per_of_old)
        out_frames.append(new_frame)
        # gray_frames.insert(i + 1, cv2.cvtColor((new_frame * 255).astype("uint8"), cv2.COLOR_RGB2GRAY))
        # frames[i][too_big_bool] = new_frame[too_big_bool]
    frames = out_frames
    dump_frames(frames, "optical_flow")
    return frames


def dump_frames(frames, folder):
    """
    Dumps the frames to the specified folder.

    Parameters:
    frames: a list of numpy arrays or PIL Image objects representing the frames.
    folder: a string representing the folder to dump the frames to.
    >>> frames = [
    ...     Image.fromarray((np.array([
    ...         [0, 0, 0],
    ...         [0, 0, 0]
    ...     ]) * 255).astype(np.uint8)),
    ...     Image.fromarray((np.array([
    ...         [0.5, 0.5, 0.5],
    ...         [0.5, 0.5, 0.5]
    ...     ]) * 255).astype(np.uint8))
    ... ]
    >>> dump_frames(frames, "test")
    >>> os.system("explorer .")
    """
    import os
    import numpy as np
    from PIL import Image
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        Image.fromarray((frame * 255).astype(np.uint8)).save(os.path.join(folder, "frame_{}.png".format(i)))


def test_limit_per_pixel_change_slice_optical_flow():
    import random
    random.seed(0)  # set a seed for reproducibility
    # generate two random images
    img1 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    frames = [img1, img2]
    max_change = 0.5
    # the first frame should not be modified
    assert np.array_equal(limit_per_pixel_change_slice_optical_flow(frames, max_change)[0], img1)
    # the second frame should have pixels that have changed more than max_change scaled
    assert not np.array_equal(limit_per_pixel_change_slice_optical_flow(frames, max_change)[1], img2)


def batched_renderer(SBIMulti, SBIMA_render_func, to_npy=False, rparam: DefaultRender = DefaultRender(),
                     early_stop=None):
    import sys
    import time
    import numpy as np
    SBIMulti = SBIMulti.combined
    images_to_save = []
    batch_times = []
    start_time = time.time()
    early_stop = sys.float_info.max if early_stop is None else early_stop

    for i in range(0, len(SBIMulti), MAX_BATCH_SIZE):
        max_batch_size = min(MAX_BATCH_SIZE, len(SBIMulti) - i)
        max_idx = min(i + MAX_BATCH_SIZE, len(SBIMulti))
        slice = SBIMulti[i:max_idx]
        slice.render.batch_size = max_batch_size
        results = SBIMA_render_func(slice.combined, 0)
        images_to_save = images_to_save + results.all_images

        batch_times.append(time.time())
        print(f"Images {i} to {i + MAX_BATCH_SIZE}, of {len(SBIMulti)}")
        if len(batch_times) > 1:
            print(f"batch time: {batch_times[-1] - batch_times[-2]}")
        if time.time() - start_time > early_stop:
            print("early stop")
            break
    if to_npy:
        images_to_save = [np.array(img).astype("float16") / 255.0 for img in images_to_save]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    return images_to_save


def batched_selective_renderer(SBIMultiArgs, SBIMA_render_func, rparam: DefaultRender = DefaultRender(),
                               early_stop=None):
    """
    >>> if True:
    ...   import PIL
    ...   import numpy as np
    ...   f = SBMultiSampleArgs(render=SBIRenderParams(),hyper=SBIHyperParams(prompt="0"))
    ...   results = lambda :None;results.all_images=[np.random.rand(512, 512, 3) for _ in range(MAX_BATCH_SIZE+1)]
    ...   for i in range(1441-1):
    ...     f += SBIHyperParams(prompt=str(i))
    ...   batched_selective_renderer(SBIMultiArgs=f, early_stop=10,SBIMA_render_func=lambda x,y:results )

    """
    import time
    import numpy as np
    from PIL import Image
    images_to_save = []
    batch_times = []
    start_time = time.time()
    # start by rendering only half of the frames
    even_SBIM = SBIMultiArgs[::2]
    odd_SBIM = SBIMultiArgs[1::2]

    even_results = batched_renderer(even_SBIM,
                                    SBIMA_render_func=SBIMA_render_func,
                                    to_npy=True,
                                    rparam=rparam,
                                    early_stop=early_stop)
    all_results = []

    for i in range(len(SBIMultiArgs)):
        if i % 2 == 0:
            try:
                idx = i // 2
                all_results.append((i,
                                    0,
                                    even_results[idx]))
            except IndexError:
                print(f"index error for {i}")
        else:
            all_results.append(None)

    for i in range(1, len(all_results), 2):
        if all_results[i] is None:
            all_results[i] = SBIMultiArgs[i]

    if len(all_results) % 2 == 0:
        all_results[-1] = (len(all_results) - 1, 0, all_results[-1])

    for i in range(0, len(all_results) - 2, 2):
        imga = all_results[i][2]
        imgb = all_results[i + 2][2]
        difference = np.mean(np.square(imga - imgb))
        all_results[i + 1] = (i + 1, difference, all_results[i + 1])

    # all_results[-1] = SBIMultiArgs[i]

    if isinstance(all_results[-1], SBMultiSampleArgs):
        all_results[-1] = (len(all_results) - 1, 1, all_results[-1])

    all_results_sorted_by_difference = \
        sorted(all_results,
               key=lambda x: np.mean(x[1]) * (isinstance(x, tuple))
               , reverse=True)
    threshold = 0.015 / 10
    new_SBIM = None
    for idx, difference, SBIMultiArg in all_results_sorted_by_difference:
        if difference > threshold:
            if new_SBIM is None:
                new_SBIM = SBIMultiArg
            else:
                new_SBIM += SBIMultiArg
        else:
            pass
    new_results = []
    if new_SBIM is not None:
        new_results = batched_renderer(new_SBIM,
                                       SBIMA_render_func=SBIMA_render_func,
                                       to_npy=True,
                                       rparam=rparam,
                                       early_stop=early_stop)

    results_zip = zip(all_results_sorted_by_difference, new_results)
    results_done = [(*i[0], i[1]) for i in results_zip]

    all_results_dict = {i[0]: i for i in all_results}
    results_done_dict = {i[0]: [i[0], i[1], i[3]] for i in results_done}
    all_results_dict.update(results_done_dict)
    to_save = [i[2] for i in sorted(all_results_dict.values(), key=lambda x: x[0]) if isinstance(i[2], np.ndarray)]
    for i in to_save:
        images_to_save.append(Image.fromarray((i * 255).astype("uint8")))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    return images_to_save


class SBMultiSampleArgs:
    """This is a class to hold and manage a collection of arguments to pass to the model
     with a batch size equal to the number of samples in the collection
     The arguments that are passed to the model and can be different per sample are
     prompt, negative_prompt, steps, seed, subseed, subseed_strength, cfg_scale
     The arguments that are passed to the model and are the same for all samples are
     width, height, restore_faces, tiling, batch_count, batch_size, sampler_index

     they have been split into hyper params and render params"""

    def __init__(self, render: SBIRenderParams, hyper):
        # this just ensures that all the params are lists
        self._hyper: SBIHyperParams = self._make_list(hyper)
        self._render = render
        self._length = len(self._hyper)
        if self._hyper == []:
            pass
        elif isinstance(self._hyper[0], str):
            self._hyper = [SBIHyperParams(prompt=p) for p in self._hyper]
        elif isinstance(self._hyper[0], SBIHyperParams):
            pass

        self.__post_init__()

    def __post_init__(self):
        self._length = len(self._hyper)

    @property
    def combined(self):
        # combines the contents of each hyper param into a single hyper param
        # this is useful for when you want to run a single render with multiple hyper params
        self._combined = copy.deepcopy(self._hyper[0])
        for i in range(1, self._length):
            self._combined += self._hyper[i]
        return SBMultiSampleArgs(render=self._render, hyper=[self._combined])

    def _make_list(self, item):

        if isinstance(item, list):
            return item
        else:
            return [item]

    def __add__(self, other):
        # if other is a tuple of hyper and render
        if isinstance(other, SBIHyperParams):
            # add the params to the lists
            self._hyper.append(other)
            # maintain the other attributes
            self.__post_init__()
        elif isinstance(other, SBMultiSampleArgs):
            self._hyper.append(other._hyper[0])
            if other._render != None:
                # if the other render params are not None
                Warning("Render params passed to add will be ignored")
            # maintain the other attributes
            self.__post_init__()
        else:
            raise TypeError("Can only add SBIHyperParams or SBMultiSampleArgs")

        return self

    @property
    def hyper(self):
        if self._length == 1:
            # if there is only one hyper param, return it
            return self._hyper[0]
        else:
            return self._hyper

    @hyper.setter
    def hyper(self, value):
        self._hyper = self._make_list(value)
        self.__post_init__()

    @property
    def render(self):
        return self._render

    def __iter__(self):
        my_iter = iter(zip(self._hyper, [self._render] * len(self._hyper)))
        return my_iter

    def __len__(self):
        return self.hyper.prompt.__len__()

    def __getitem__(self, item):
        tmp = SBMultiSampleArgs(render=self._render, hyper=self._hyper[0][item])
        return tmp


@dataclass
class CommonMembers:
    batch_size: int = 1
    processed: Optional[Any] = None
    all_images: List = field(default_factory=list)
    all_prompts: List = field(default_factory=list)
    all_seeds: List = field(default_factory=list)
    all_subseeds: List = field(default_factory=list)
    all_negative_prompts: List = field(default_factory=list)
    all_steps: List = field(default_factory=lambda: [0])
    all_subseed_strengths: List = field(default_factory=lambda: [0.0])
    all_cfg_scales: List = field(default_factory=lambda: [0.0])
    cfg_scale: float = 0.0
    height: int = 256
    width: int = 256
    negative_prompt: str = ""
    sampler_name: str = "default_sampler"
    steps: int = 0
    sb_iparams: Optional[Any] = None
    img_hyper_params_list: List = field(default_factory=list)
    all_as_stb_image_params: List = field(default_factory=list)


class SBImageResults(CommonMembers):
    """
    This class is used to hold the results of a render provided in/by the modules.processing.Processed class
    """

    def __init__(self, processed=None, api_results=None):
        super().__init__()
        if processed is not None:
            self.batch_size = processed.batch_size
            self.processed = processed
            self.all_images = processed.images
            self.all_prompts = processed.all_prompts
            self.all_seeds = processed.all_seeds
            self.all_subseeds = processed.all_subseeds
            if len(self.all_subseeds) != len(self.all_seeds):
                self.all_subseeds = self.all_seeds.copy()

            # these need to be adapted to be changable inter-batch if possible
            self.all_negative_prompts = [processed.negative_prompt] * processed.batch_size
            self.all_steps = [processed.steps] * processed.batch_size
            self.all_subseed_strengths = [processed.subseed_strength] * len(processed.all_seeds)
            self.all_cfg_scales = [processed.cfg_scale] * len(processed.all_seeds)

            self.batch_size = processed.batch_size
            self.cfg_scale = processed.cfg_scale
            self.clip_skip = processed.clip_skip
            self.height = processed.height
            self.width = processed.width
            self.job_timestamp = processed.job_timestamp
            self.negative_prompt = processed.negative_prompt
            self.sampler_name = processed.sampler_name
            self.steps = processed.steps
            self.sb_iparams: SBMultiSampleArgs = self.sb_multi_sample_args_from_sd_results()

        elif api_results is not None:
            self.batch_size = api_results['batch_size']
            self.processed = api_results
            self.all_images = api_results['images']
            self.all_prompts = api_results['prompt']
            self.all_seeds = api_results['seed']
            self.all_subseeds = api_results['subseed']
            if len(self.all_subseeds) != len(self.all_seeds):
                self.all_subseeds = self.all_seeds.copy()

            # these need to be adapted to be changable inter-batch if possible
            self.all_negative_prompts = [api_results['negative_prompt']] * api_results['batch_size']
            self.all_steps = [api_results['steps']] * api_results['batch_size']
            self.all_subseed_strengths = [api_results['subseed_strength']] * api_results['batch_size']
            self.all_cfg_scales = [api_results['cfg_scale']] * api_results['batch_size']

            self.batch_size = api_results['batch_size']
            self.cfg_scale = api_results['cfg_scale']
            # self.clip_skip = api_results['clip_skip']
            self.height = api_results['height']
            self.width = api_results['width']
            # self.job_timestamp = api_results['job_timestamp']
            self.negative_prompt = api_results['negative_prompt']
            self.sampler_name = api_results['sampler_name']
            self.steps = api_results['steps']
            self.sb_iparams: SBMultiSampleArgs = self.sb_multi_sample_args_from_sd_results()

        self.img_hyper_params_list = [SBIHyperParams(prompt=prompt,
                                                     negative_prompt=negative_prompt,
                                                     steps=steps,
                                                     seed=seed,
                                                     subseed=subseed,
                                                     subseed_strength=subseed_strength,
                                                     cfg_scale=cfg_scale)
                                      for prompt,
                                          negative_prompt,
                                          steps,
                                          seed,
                                          subseed,
                                          subseed_strength,
                                          # todo: fix the dimensionality of the ones that are not lists
                                          cfg_scale in zip(self.all_prompts,
                                                           self.all_negative_prompts,
                                                           self.all_steps,
                                                           self.all_seeds,
                                                           self.all_subseeds,
                                                           self.all_subseed_strengths,
                                                           self.all_cfg_scales)]

        tmp_list_of_st_sq_render_params = [SBIRenderParams(width=self.width,
                                                           height=self.height,
                                                           # restore_faces=self.restore_faces,
                                                           # tiling does not make it to through the conversion to "procesed"
                                                           # tiling=processed.extra_generation_params[""],
                                                           tiling=None,
                                                           # batch count does not make it to through the conversion to "procesed"
                                                           # batch_count=processed.,batch_count
                                                           batch_count=None,
                                                           batch_size=self.batch_size,
                                                           sampler_name=self.sampler_name)
                                           for _ in range(self.batch_size)]

        self.all_as_stb_image_params = [SBMultiSampleArgs(hyper=hyper,
                                                          render=render)
                                        for hyper, render in zip(self.img_hyper_params_list,
                                                                 tmp_list_of_st_sq_render_params)]

    def __iter__(self):
        o = iter(self.img_hyper_params_list)
        return o

    def __getitem__(self, item):
        return self.img_hyper_params_list[item]

    def __add__(self, other):
        if isinstance(other, SBImageResults):
            for key in self.__dict__.keys():
                my_value = self.__dict__[key]
                other_value = other.__dict__[key]
                if not isinstance(my_value, list):
                    my_value = [my_value]
                if not isinstance(other_value, list):
                    other_value = [other_value]

                self.__dict__[key] = my_value + other_value

            return self
        else:
            print("Cannot add SBImageResults to non SBImageResults")

    def sb_multi_sample_args_from_sd_results(self) -> SBMultiSampleArgs:
        # convert the StableDiffusionProcessingTxt2Img params to SBMultiSampleArgs
        processed = self.processed
        if len(self.all_seeds) == len(self.all_prompts):
            # then these results are for multiple seeds and prompts
            t_hyp = SBIHyperParams(prompt=self.all_prompts,
                                   negative_prompt=self.negative_prompt,
                                   steps=self.steps,
                                   seed=self.all_seeds,
                                   subseed=self.all_subseeds,
                                   # TODO: check if this is valid for multiple subseed
                                   #  strengths
                                   subseed_strength=self.all_subseed_strengths,
                                   cfg_scale=self.cfg_scale)
        else:
            t_hyp = SBIHyperParams(prompt=self.all_prompts,
                                   negative_prompt=self.negative_prompt,
                                   steps=self.steps,
                                   seed=self.all_seeds,
                                   subseed=self.all_subseeds,
                                   subseed_strength=self.all_subseed_strengths,
                                   cfg_scale=self.cfg_scale)

        t_render = SBIRenderParams(width=self.width,
                                   height=self.height,
                                   # restore_faces=self.,
                                   # TODO: figure out where to get this from
                                   tiling=False,
                                   # TODO: check if the Processed class is just for one batch always
                                   batch_count=1,
                                   batch_size=self.batch_size,
                                   sampler_name=self.sampler_name)

        t_params = SBMultiSampleArgs(hyper=t_hyp, render=t_render)

        return t_params


def get_frame_deltas(frames: List[Image.Image]) -> List[float]:
    """"get the difference between each image"""
    frame_deltas = []
    for i in range(len(frames) - 1):
        frame_delta = get_img_diff(frames[i], frames[i + 1])
        frame_deltas.append(frame_delta)
    return frame_deltas


def get_linear_interpolation(param: PIL.Image, param1: PIL.Image, t: float):
    """linear interpolation between two images"""
    r_img = Image.blend(param, param1, t)
    r_img.info = {"frame_type": "i"}
    return r_img


def save_tmp_images_for_debug(all_imgs_by_seconds, loop_count):
    for k, v in all_imgs_by_seconds.items():
        pth = os.path.join(STORYBOARD_TMP_PATH, f"tmp_{loop_count}")
        # create the directory if it doesn't exist
        if not os.path.exists(pth):
            os.mkdir(pth)

        ft = v.info["frame_type"] if "frame_type" in v.info else "n"
        fn = f"tmp_{ft}_{k}"
        fext = "png"

        full_pth = os.path.join(pth, f"tmp_{ft}_{k}.png")

        from modules.images import save_image
        # v.save(full_pth)
        save_image(image=v,
                   basename=fn,
                   path=pth,
                   forced_filename=fn,
                   extension=fext,
                   existing_info=v.info,
                   )


def compose_storyboard_render(my_render_params, all_state, early_stop, storyboard_params, test,
                              test_render, ui_params, SBIMA_render_func, base_SBIMulti: SBMultiSampleArgs):
    # in the interest of syncing the legth of the audio voice over and the length of the video it is important to
    # consider the length of the audio first, primarily because the audio is much quicker to render, but also
    # because it is harder to manipulate temporaly then the mostly arbitrary contents of the video

    # the audio is rendered first, attempting to reach some target length.
    # the video is rendered second, to match the length of the resultant audio
    my_render_params = copy.deepcopy(my_render_params)
    mytext = ui_params[0]
    if test or test_render:
        mytext = "one two three four five six seven eight nine ten"
    audio_f_path, vo_len_secs = create_voice_over_for_storyboard(mytext, 1, DefaultRender.seconds)

    my_render_params.num_frames_per_section = int((my_render_params.fps * vo_len_secs) / my_render_params.sections)
    my_render_params.num_frames = my_render_params.num_frames_per_section * my_render_params.sections
    my_render_params.seconds = vo_len_secs
    if my_render_params.num_frames < my_render_params.min_frames_per_render:
        my_render_params.num_frames = my_render_params.min_frames_per_render
        my_render_params.fps = my_render_params.num_frames / my_render_params.seconds
        my_render_params.frames_per_section = int(my_render_params.num_frames / my_render_params.sections)
    else:
        use_fps = my_render_params.fps

    prompt_sections = get_frame_values_for_prompt_word_weights(
        [params.prompt for params in storyboard_params],
        num_frames=my_render_params.num_frames_per_section
    )
    #  turn the weights into a list of prompts
    prompts = []
    for section in prompt_sections:
        for frame in section:
            prompts.append(" ".join([f"({word}:{weight})" for word, weight in frame]))
    seeds = get_frame_seed_data(storyboard_params, my_render_params.num_frames_per_section)
    # create a base SBIRenderArgs object
    # feature: this should allow the user to change the settings for rendering

    # base_SBIMulti:SBMultiSampleArgs = StoryBoardGradio.get_sb_multi_sample_params_from_ui(ui_params)
    base_Hyper = copy.deepcopy(base_SBIMulti.hyper)
    # turn the list of prompts and seeds into a list of CallArgsAsData using the base_params as a template
    # populate the storyboard_call_multi with the prompts and seeds
    out_hyper = SBIHyperParams()
    for prompt, seed in zip(prompts, seeds):
        out_hyper += SBIHyperParams(
            prompt=prompt,
            seed=seed[0],
            subseed=seed[1],
            subseed_strength=seed[2],
            negative_prompt=base_Hyper.negative_prompt,
            steps=base_Hyper.steps,
            cfg_scale=base_Hyper.cfg_scale,
        )
    out_SBM = SBMultiSampleArgs(hyper=out_hyper, render=base_SBIMulti.render)
    images_to_save = batched_selective_renderer(out_SBM,
                                                rparam=my_render_params,
                                                early_stop=early_stop,
                                                SBIMA_render_func=SBIMA_render_func)

    target_mp4_f_path = compose_file_handling(audio_f_path, images_to_save)
    return all_state, target_mp4_f_path


def compose_file_handling(audio_f_path,
                          images_to_save,
                          fps=DefaultRender.fps,
                          width=DefaultRender.width,
                          height=DefaultRender.height):
    working_dir = os.path.join(STORYBOARD_RENDER_PATH, "tmp")
    print(f'working_dir: {working_dir}')
    print(f'audio_f_path: {audio_f_path}')

    video_f_path = make_mp4_from_images(
        images_to_save,
        working_dir,
        "video.mp4", width, height, False,
        fps=fps,
        filter_func=lambda x: limit_per_pixel_change_slice(x, .5))

    print(f'video_f_path: {video_f_path}')
    complete_mp4_f_path = join_video_audio(video_f_path, audio_f_path)
    target_mp4_f_path = os.path.join(STORYBOARD_RENDER_PATH,
                                     f"StoryBoard-{str(random.randint(1, 1000000))}.mp4")
    print(f"target_mp4_f_path: {target_mp4_f_path}")
    # delete storyboard.mp4
    os.remove(video_f_path)
    # delete the audio file
    os.remove(audio_f_path)
    # move the mp4 to the storyboard folder using os.renam
    os.rename(complete_mp4_f_path, target_mp4_f_path)
    return target_mp4_f_path
