import copy
import json

from dataclasses import dataclass
from typing import List, Any

import numpy


DEV_MODE = True
DEFAULT_HYPER_PARAMS = {
    "prompt": "",
    "negative_prompt": "",
    "steps": 8,
    "seed": 0,
    "subseed": 0,
    "subseed_strength": 0,
    "cfg_scale": 7,
}



@dataclass
class DEFAULT_HYPER_PARAMS:
    prompt: str = DEFAULT_HYPER_PARAMS["prompt"]
    negative_prompt: str = DEFAULT_HYPER_PARAMS["negative_prompt"]
    steps: int = DEFAULT_HYPER_PARAMS["steps"]
    seed: int = DEFAULT_HYPER_PARAMS["seed"]
    subseed: int = DEFAULT_HYPER_PARAMS["subseed"]
    subseed_strength: int = DEFAULT_HYPER_PARAMS["subseed_strength"]
    cfg_scale: int = DEFAULT_HYPER_PARAMS["cfg_scale"]

DEV_HYPER_PARAMS = DEFAULT_HYPER_PARAMS(steps=1)

DEFAULT_HYPER_PARAMS = DEV_HYPER_PARAMS
GTTS_SAMPLE_RATE = 24000.0
MAX_BATCH_SIZE = 10

#DEV_HYPER_PARAMS["steps"] = 1

if DEV_MODE:
    pass

@dataclass
class DefaultRender:
    fps: int = 15
    minutes: int = 2
    seconds: int = minutes * 60
    sections: int = 2
    num_frames = int(seconds * fps)
    min_frames_per_render = fps * 10
    num_frames_per_sctn = int(num_frames / sections)
    early_stop_seconds = int(60 * 30 * 16) # 8 hours
    width = 512
    height = 512
    restore_faces = False
    tiling = False
    batch_count = 1
    batch_size = MAX_BATCH_SIZE
    sampler_index = 9




class SBIHyperParams:
    """
    the idea with this class is to provide a useful interface for the user to set,add,index the hyper parameters
    """

    def __init__(self, negative_prompt=DEFAULT_HYPER_PARAMS.negative_prompt,
                 steps=DEFAULT_HYPER_PARAMS.steps,
                 seed=DEFAULT_HYPER_PARAMS.seed,
                 subseed=DEFAULT_HYPER_PARAMS.subseed,
                 subseed_strength=DEFAULT_HYPER_PARAMS.subseed_strength,
                 cfg_scale=DEFAULT_HYPER_PARAMS.cfg_scale,
                 prompt=DEFAULT_HYPER_PARAMS.prompt,
                 **kwargs):
        # this just ensures that all the params are lists

        # If prompt was not passed via init, then check kwargs, else raise value error
        # this is so that the prompt can be passed as a positional argument or a keyword argument
        # so that the class can be created by unpacking the dictionary of this object
        if prompt is None:
            if "_prompt" in kwargs:
                _prompt = kwargs["_prompt"]
            else:
                raise ValueError("prompt is required")
        else:
            _prompt = prompt

        self._prompt = self._make_list(_prompt)
        self.negative_prompt = self._make_list(negative_prompt)
        self.steps = self._make_list(steps)
        self.seed = self._make_list(seed)
        self.subseed = self._make_list(subseed)
        self.subseed_strength = self._make_list(subseed_strength)
        self.cfg_scale = self._make_list(cfg_scale)

    def __getitem__(self, item):
        return SBIHyperParams(prompt = self._prompt[item], negative_prompt=self.negative_prompt[item],
                              steps=self.steps[item], seed=self.seed[item], subseed=self.subseed[item],
                              subseed_strength=self.subseed_strength[item], cfg_scale=self.cfg_scale[item])
    def _make_list(self, item: object) -> [object]:
        if isinstance(item, list):
            return item
        else:
            return [item]

    def __add__(self, other):
        # this allows this class to be used to append to itself
        if isinstance(other, SBIHyperParams):
            return SBIHyperParams(
                prompt=self._prompt + other._prompt,
                negative_prompt=self.negative_prompt + other.negative_prompt,
                steps=self.steps + other.steps,
                seed=self.seed + other.seed,
                subseed=self.subseed + other.subseed,
                subseed_strength=self.subseed_strength + other.subseed_strength,
                cfg_scale=self.cfg_scale + other.cfg_scale,
            )

    @property
    def prompt(self):
        # if there is only one prompt then return it as a string
        if len(self._prompt) == 1:
            return self._prompt[0]
        else:
            return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = self._make_list(value)
    def __json__(self):
        # serialize the object to a json string
        return json.dumps(self.__dict__)

    def __str__(self):
        return self.__json__()


@dataclass
class SBIRenderParams:
    width: int = 512
    height: int = 512
    restore_faces: bool = False
    tiling: bool = False
    batch_count: int = 1
    batch_size: int = MAX_BATCH_SIZE
    sampler_index: int = 9

class SBMultiSampleArgs:
    """This is a class to hold and manage a collection of arguments to pass to the model
     with a batch size equal to the number of samples in the collection
     The arguments that are passed to the model and can be different per sample are
     prompt, negative_prompt, steps, seed, subseed, subseed_strength, cfg_scale
     The arguments that are passed to the model and are the same for all samples are
     width, height, restore_faces, tiling, batch_count, batch_size, sampler_index

     they have been split into hyper params and render params"""

    def __init__(self, render: SBIRenderParams, hyper: List[SBIHyperParams] = []):
        # this just ensures that all the params are lists
        self._hyper = self._make_list(hyper)
        self._render = render
        self._length = len(self._hyper)
        for i in range(self._length):
            if not isinstance(self._hyper[i], SBIHyperParams):
                raise TypeError(f'item {i} in hyper is not an instance of SBIHyperParams')
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
        return self._length

    def __getitem__(self, item):
        tmp = SBMultiSampleArgs(render=self._render, hyper=self._hyper[item])
        return tmp

class SBImageResults:
    """
    This class is used to hold the results of a render provided in/by the modules.processing.Processed class
    """

    def __init__(self, processed):
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
        self.sampler_index = processed.sampler_index
        self.sampler_name = processed.sampler
        self.steps = processed.steps
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
                                                           restore_faces=processed.restore_faces,
                                                           # tiling does not make it to through the conversion to "procesed"
                                                           # tiling=processed.extra_generation_params[""],
                                                           tiling=None,
                                                           # batch count does not make it to through the conversion to "procesed"
                                                           # batch_count=processed.,batch_count
                                                           batch_count=None,
                                                           batch_size=processed.batch_size,
                                                           sampler_index=processed.sampler_index)
                                           for _ in range(processed.batch_size)]

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
            self.img_hyper_params_list += other.img_hyper_params_list
        else:
            print("Cannot add SBImageResults to non SBImageResults")

    def sb_multi_sample_args_from_sd_results(self) -> SBMultiSampleArgs:
        # convert the StableDiffusionProcessingTxt2Img params to SBMultiSampleArgs
        processed = self.processed
        if len(processed.all_seeds) == len(processed.all_prompts):
            # then these results are for multiple seeds and prompts
            t_hyp = SBIHyperParams(prompt=processed.all_prompts,
                                   negative_prompt=processed.negative_prompt,
                                   steps=processed.steps,
                                   seed=processed.all_seeds,
                                   subseed=processed.all_subseeds,
                                   # TODO: check if this is valid for multiple subseed
                                   #  strengths
                                   subseed_strength=processed.subseed_strength,
                                   cfg_scale=processed.cfg_scale)
        else:
            t_hyp = SBIHyperParams(prompt=processed.prompt,
                                   negative_prompt=processed.negative_prompt,
                                   steps=processed.steps,
                                   seed=processed.seed,
                                   subseed=processed.subseed,
                                   subseed_strength=processed.subseed_strength,
                                   cfg_scale=processed.cfg_scale)

        t_render = SBIRenderParams(width=processed.width,
                                   height=processed.height,
                                   restore_faces=processed.restore_faces,
                                   # TODO: figure out where to get this from
                                   tiling=False,
                                   # TODO: check if the Processed class is just for one batch always
                                   batch_count=1,
                                   batch_size=processed.batch_size,
                                   sampler_index=processed.sampler_index)

        t_params = SBMultiSampleArgs(hyper=t_hyp, render=t_render)

        return t_params

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


def join_video_audio(video_file, audio_file):
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
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
    ...  create_voice_over_for_storyboard("one two three four five six seven eight nine ten ", None, 10)
    ...  break
    """
    # time everything

    print("on_preview_audio")
    save_sample_rate = 44100
    # get the audio samples to aud_out while timing it

    t, (aud_out, aud_length_secs) = quick_timer(get_samples_from_gtts, text_to_read)
    print(f"get_samples_from_gtts: latency {t}")
    if aud_length_secs < vo_length_sec:
        t, (aud_out_slow, aud_length_secs_slow) = quick_timer(get_samples_from_gtts, text_to_read, slow=True)
    reg_dist = (vo_length_sec - aud_length_secs) ** 2
    slow_dist = (vo_length_sec - aud_length_secs_slow) ** 2
    if slow_dist < reg_dist:
        aud_out = aud_out_slow
        aud_length_secs = aud_length_secs_slow

    t, (data, data_sample_rate) = quick_timer(robot_voice_effect, aud_out)
    print(f'robot_voice_effect latency: {t}')

    audio_length_secs = max(*data.shape) / data_sample_rate
    if audio_length_secs > vo_length_sec:
        # remove silence of longer than .5 seconds
        t, data = quick_timer(remove_silence, data, data_sample_rate, vo_length_sec)
        print(f'remove_silence latency: {t}')

    audio_length_secs = max(*data.shape) / data_sample_rate

    rnd_file = write_mp3(data, data_sample_rate, save_sample_rate)

    return rnd_file, audio_length_secs


def write_mp3(effected, data_sample_rate, save_sample_rate):
    import numpy as np
    import random
    # import AudioArrayClip from moviepy
    from moviepy.audio.AudioClip import AudioArrayClip

    rnd_file = f"tmp_{random.randint(0, 1000000)}.wav"
    afc = AudioArrayClip(np.moveaxis(effected, 0, -1), data_sample_rate)
    afc.write_audiofile(rnd_file, fps=save_sample_rate)
    return rnd_file


def robot_voice_effect(aud_out):
    import numpy as np

    rate = 44100
    print(aud_out.shape)
    effected = np.copy(aud_out)
    divisor = 4
    # effected = board(effected, sample_rate=rate)
    effected = effected[:, ::divisor]
    offset = int(rate / 125)
    effected = effected[:, :effected.shape[1] - offset] + effected[:, offset:]
    effected = effected[:, :effected.shape[1] - offset] + effected[:, offset:]
    effected = effected[:, :effected.shape[1] - offset] + effected[:, offset:]
    effected = effected[:, :effected.shape[1] - offset] + effected[:, offset:]
    # normalize
    effected = effected / np.max(np.abs(effected))
    print(effected.shape)
    rate = int(rate / divisor)

    return effected, rate


def get_samples_from_gtts(text_to_read, slow=False) -> (numpy.ndarray, float):
    # TODO: this needs paraellization see https://gtts.readthedocs.io/en/latest/tokenizer.html#minimizing
    # 100 characters per request

    from gtts import gTTS
    from pedalboard.io import AudioFile
    import moviepy.editor as mpy
    import os

    audio = gTTS(
        text=text_to_read,
        lang="en",
        slow=slow,
    )
    audio.save("tmp.mp3")

    mpy.AudioFileClip("tmp.mp3").write_audiofile("tmp.wav")
    with AudioFile("tmp.wav", "r") as f:
        aud_out = f.read(f.frames)
    # get the length of the audio
    aud_length_secs = aud_out.shape[1] / GTTS_SAMPLE_RATE

    # delete the files
    os.remove("tmp.mp3")
    os.remove("tmp.wav")

    return aud_out, aud_length_secs


def make_mp4(input_path, filepath, filename, width, height, keep, fps=30) -> str:
    # TODO: use moviepy instead
    import subprocess
    import os
    import glob
    import uuid
    image_input_path = os.path.join(input_path, "%05d.png")
    mp4_path = os.path.join(filepath, f"{str(filename)}.mp4")
    # check if the file exists, if it does, change mp4_path to include part of a uuid
    if os.path.exists(mp4_path):
        print(f"file exists, changing name to {mp4_path}")
        mp4_path = os.path.join(filepath, f"{str(filename)}_{str(uuid.uuid4()).split('-')[-1]}.mp4")
    mp4_path = mp4_path.split(".mp4")[0] + ".mp4"
    # make the mp4

    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', str(image_input_path),
        '-c:v', 'libx264',
        '-vf', 'scale=' + str(width) + ':' + str(height),
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
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
                         temporal_noise_filter=True) -> str:
    import os
    from PIL import Image
    """Make an mp4 from a list of images, using make_mp4"""

    if temporal_noise_filter:
        image_list = limit_per_pixel_change_slice(image_list, 0.0333)
        # convert numpy array to PIL image
        image_list = [Image.fromarray((image * 255).astype("uint8")) for image in image_list]

    # save the images to a temp folder
    temp_folder_path = os.path.join(filepath, "temp")
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

    return make_mp4(f"{temp_folder_path}", filepath, filename, width, height, keep, fps=DefaultRender.fps)


def get_word_weight_at_percent(section, word_index, percent):
    """
    >>> get_word_weight_at_percent([("dog",0.0),("cat",1.0)],0,0.5)
    0.5
    """
    start_weight = section[0][word_index][1]
    end_weight = section[1][word_index][1]
    return start_weight + percent * (end_weight - start_weight)


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


def get_frame_seed_data(board_params, _num_frames) -> [()]:  # List[(seed,subseed,weight)]
    """
    interpolation between seeds is done by setting the subseed to the target seed of the next section, and then
    interpolating the subseed weight from 0  to 1
    """
    sections = [
        [board_params[0], board_params[1]],
        [board_params[1], board_params[2]],
    ]
    all_frames = []
    for start, end in sections:
        seed = start.seed
        subseed = end.seed
        for i in range(_num_frames):
            sub_seed_weight = i / (_num_frames - 1)
            all_frames.append((seed, subseed, sub_seed_weight))
    return all_frames


def get_prompt_words_and_weights_list(prompt) -> List[List[str]]:
    """
    >>> get_prompt_words_and_weights_list("hello:1 world:.2 how are (you:1.0)")
    [('hello', 1.0), ('world', 0.2), ('how', 1.0), ('are', 1.0), ('you', 1.0)]
    """
    prompt = sanitize_prompt(prompt)
    words = prompt.split(" ")
    possible_word_weight_pairs = [i.split(":") for i in words]
    w = 1.0
    out: list[tuple[Any, float]] = []
    for word_weight_pair in possible_word_weight_pairs:
        value_count = len(word_weight_pair)  # number of values in the tuple
        # if the length of the item that is possibly a word weight pair is 1 then it is just a word
        if value_count == 1:  # when there is no weight assigned to the word
            w = 1.0
        # if the length of the item that is possibly a word weight pair is 2 then it is a word and a weight
        elif value_count == 2:  # then there is a word and probably a weight in the tuple
            # if the second item in the word weight pair is a float then it is a weight
            try:
                w = float(word_weight_pair[1])
            # if the second item in the word weight pair is not a float then it is not a weight
            except ValueError:
                raise ValueError(f"Could not convert {word_weight_pair[1]} to a float")
        else:
            raise ValueError(f"Could not convert {word_weight_pair} to a word weight pair")
        out.append((word_weight_pair[0], w))
    return out


def sanitize_prompt(prompt):
    prompt = prompt.replace(",", " ").replace(". ", " ").replace("?", " ").replace("!", " ").replace(";", " ")
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("\r", " ")
    prompt = prompt.replace("[", " ").replace("]", " ")
    prompt = prompt.replace("{", " ").replace("}", " ")
    # compact blankspace
    for i in range(10):
        prompt = prompt.replace("  ", " ")

    prompt = prompt.replace("(", "").replace(")", "")
    return prompt.strip()


def limit_per_pixel_change_slice(frames, max_change):
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


@staticmethod
def batched_renderer(SBIMulti, SBIMA_render_func, to_npy=False, rparam: DefaultRender = DefaultRender(),
                     early_stop=None):
    import time
    import numpy as np
    images_to_save = []
    batch_times = []
    start_time = time.time()
    for i in range(0, len(SBIMulti), MAX_BATCH_SIZE):
        max_batch_size = min(MAX_BATCH_SIZE, len(SBIMulti) - i)
        max_idx = min(i + MAX_BATCH_SIZE, len(SBIMulti))
        slice = SBIMulti[i:max_idx]
        slice.render.batch_size = max_batch_size
        results = SBIMA_render_func(slice.combined, 0)
        images_to_save = images_to_save + results.all_images[1:1 + len(slice)]

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


@staticmethod
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
    images_to_save = {}
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
    SBIMultiArgs = SBIMultiArgs[:len(even_results) * 2]
    # now render the other half of the frames if the difference between the two is greater than a threshold
    threshold = 0.015 / 2
    # threshold = 0.0
    to_process = None
    # first build a list of the indices of the SBIMultiArgs that need to be rendered based on the difference in the odd frames
    # need something like [(odd_img_prev,even_idx_now,odd_img_next)]
    for i in range(1, len(even_results) - 1):
        difference = np.mean(np.square(even_results[i] - even_results[i + 1]))
        print(difference)
        if difference > threshold:
            if to_process is None:
                to_process = [(i * 2, odd_SBIM[i])]
            else:
                to_process.append((i * 2, odd_SBIM[i]))
    # create the new SBIMultiArgs to render
    new_SBIM = None
    time_idxs = []
    if to_process is not None:
        print(
            f'selective renderer found {len(to_process)} frames to process of {int(len(SBIMultiArgs) / 2)} possible frames')
        for time_idx, sbim in to_process:
            if new_SBIM is None:
                new_SBIM = copy.deepcopy(sbim)
                time_idxs.append(time_idx)
            else:
                new_SBIM += copy.deepcopy(sbim)
                time_idxs.append(time_idx)

    # render the new SBIMultiArgs
    ti_sm_res = {}
    if len(time_idxs) > 0:
        smoothing_results = batched_renderer(new_SBIM,
                                             SBIMA_render_func=SBIMA_render_func,
                                             to_npy=True,
                                             rparam=rparam,
                                             early_stop=early_stop)
        ti_sm_res = dict(zip(time_idxs, smoothing_results))

    images_to_save = []
    for time_idx in range(len(SBIMultiArgs)):
        odd = time_idx % 2 == 0
        even = not odd
        if even:
            try:
                images_to_save.append(even_results[int(time_idx / 2)])
            except:
                print(f'error with time_idx {time_idx} and tmp_results {len(even_results)}')
        if odd:
            if time_idx in ti_sm_res.keys():
                images_to_save.append(ti_sm_res[time_idx])
            else:
                try:
                    images_to_save.append(even_results[int(time_idx / 2)])
                except:
                    print(f'error with time_idx {time_idx}')

    # convert the images to PIL images
    images_to_save = [Image.fromarray(np.uint8(img * 255.0)) for img in images_to_save]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    return images_to_save




