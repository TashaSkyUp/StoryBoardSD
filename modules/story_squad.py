import gradio.components
import gradio as gr
import json

from dataclasses import dataclass

print(__name__)
from typing import List, Any
keys_for_ui_in_order = ["prompt", "negative_prompt", "steps", "sampler_index", "width", "height", "restore_faces",
                         "tiling", "batch_count", "batch_size",
                         "seed", "subseed", "subseed_strength", "cfg_scale"]
MAX_BATCH_SIZE = 9
DEFAULT_HYPER_PARAMS = {
    "prompt": "",
    "negative_prompt": "",
    "steps": 8,
    "seed": -1,
    "subseed": 0,
    "subseed_strength": 0,
    "cfg_scale": 7,
}

if __name__ != "__main__" and __name__ != "story_squad":
    import random
    import doctest
    import numpy as np
    from typing import List
    import copy
    from collections import OrderedDict
    import modules
    from modules.processing import StableDiffusionProcessingTxt2Img
else:
    # TODO: figure out how to load the correct modules when running this file directly for doctests
    print("Running doctests")

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


def random_noise_image():
    return np.random.rand(512, 512, 3)


def make_gr_label(label):
    return gr.HTML(f"<div class = 'flex ont-bold text-2xl items-center justify-center'> {label} </div>")


class SBIHyperParams:
    """
    the idea with this class is to provide a useful interface for the user to set,add,index the hyper parameters
    """

    def __init__(self, negative_prompt, steps, seed, subseed, subseed_strength, cfg_scale, prompt=None, **kwargs):
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
    width: int
    height: int
    restore_faces: bool
    tiling: bool
    batch_count: int
    batch_size: int
    sampler_index: int


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
            self._hyper.append(other._hyper)
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

    def __init__(self, processed: modules.processing.Processed):
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

        # self.results_as_list= list(zip(self.all_images,

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


class StorySquad:
    def __init__(self):
        # import some functionality from the provided webui
        from modules.ui import setup_progressbar, create_seed_inputs

        # import custom sampler caller and assign it to be easy to access
        from modules.storyboard import storyboard_call_multi as storyboardtmp
        self.storyboard = storyboardtmp

        # assign imported functionality to be easy to access
        self.setup_progressbar = setup_progressbar
        self.create_seed_inputs = create_seed_inputs

        # initial state assignment
        self.all_components = {}
        self.all_state = {}

        # some functions in the AUTOMATIC1111's code base require things in a specific order
        # this list will hold the values and be populated when the ui is defined
        self.ordered_list_of_param_inputs = []

        # populate all_state with the default values
        # its hard to explain why this is done this way, but it is
        # basically gradio needs to be informed this way that the objects exist
        with self.get_story_squad_ui() as wrapped_ui:
            self.StSqUI = wrapped_ui
            self.get_story_squad_ui = lambda: self.StSqUI

            self.all_state["history"] = []
            self.all_state["im_explorer_hparams"] = []
            self.all_state["story_board"] = []

            self.all_state = gr.State(self.all_state)
            self.setup_story_board_events()

    def render_storyboard(self, *args):
        """
        >>> StorySquad.render_storyboard([CallArgsAsData(prompt= "(dog:1) cat:0",seed=1),CallArgsAsData(prompt= "(dog:1) cat:1",seed=2),CallArgsAsData(prompt= "(dog:0) cat:1",seed=3)],test=True)
        render_storyboard
        ()
        test
        [prompt: (dog:1.0) (cat:0.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 1,subseed: 2,subseed_strength: None,cfg_scale: None,sub_seed_weight: 0.0, prompt: (dog:1.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 1,subseed: 2,subseed_strength: None,cfg_scale: None,sub_seed_weight: 1.0, prompt: (dog:1.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 2,subseed: 3,subseed_strength: None,cfg_scale: None,sub_seed_weight: 0.0, prompt: (dog:0.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 2,subseed: 3,subseed_strength: None,cfg_scale: None,sub_seed_weight: 1.0]
        """
        print("render_storyboard")

        all_state = args[0]
        ui_params = args[1:]

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
            for start, end in sections:
                word_frame_weights = []
                for i in range(num_frames):
                    frame_weights = []
                    for word_idx, word_at_pos in enumerate(start):
                        word_start_weight = word_at_pos[1]
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

        storyboard_params = all_state["story_board"]
        test_render = False
        if all(map(lambda x: x is None, storyboard_params) or len(storyboard_params) == 0):
            """Test that is ran when the storyboard_params are not set but the user presses the render button"""
            test = True
            test_render = True
            test_data = [
                "(dog:1) cat:0",
                "(dog:1) cat:1",
                "(dog:0) cat:1",
            ]
            storyboard_params = [SBIHyperParams(
                prompt=prompt,
                seed=i,
                negative_prompt="",
                cfg_scale=7,
                steps=5,
                subseed=-1,
                subseed_strength=0.0,
            ) for i, prompt in enumerate(test_data)]

        else:
            test = False
            storyboard_params = all_state["story_board"]
            ui_params = args[1:]

        if test:
            print("test")
            num_frames = 2
        else:
            num_frames = 120

        prompt_sections = get_frame_values_for_prompt_word_weights(
            [params.prompt for params in storyboard_params],
            num_frames=num_frames
        )

        #  turn the weights into a list of prompts
        prompts = []
        for section in prompt_sections:
            for frame in section:
                prompts.append(" ".join([f"({word}:{weight})" for word, weight in frame]))

        seeds = get_frame_seed_data(storyboard_params, num_frames)

        # create a base SBIRenderArgs object
        # feature: this should allow the user to change the settings for rendering
        base_SBIMulti = self.get_sb_multi_sample_params_from_ui(ui_params)

        base_Hyper = copy.deepcopy(base_SBIMulti.hyper)
        # turn the list of prompts and seeds into a list of CallArgsAsData using the base_params as a template

        for prompt, seed in zip(prompts, seeds):
            base_SBIMulti += SBIHyperParams(
                prompt=prompt,
                seed=seed[0],
                subseed=seed[1],
                subseed_strength=seed[2],
                negative_prompt=base_Hyper.negative_prompt,
                steps=base_Hyper.steps,
                cfg_scale=base_Hyper.cfg_scale,
            )

        images_to_save = []

        if not test or test_render:
            for i in range(0, len(base_SBIMulti), MAX_BATCH_SIZE):
                # render the images
                slice = base_SBIMulti[i:i + MAX_BATCH_SIZE]
                results = self.storyboard(slice.combined, 0)
                images_to_save.append(results.all_images)

        # return base_SBIMulti, images_to_save

    def update_image_exp_text(self, h_params:List[SBIHyperParams]):
        print("update_image_exp_text")
        print(h_params)
        o = [str(i) for i in h_params]
        return o

    def render_explorer(self, params: SBMultiSampleArgs):
        # calls self.storyboard with the params
        # returns a list of images, and a list of params
        # this is the function that is called by the image explorer
        # it is called with a SBMultiSampleArgs object

        results: SBImageResults = self.storyboard(params, 0)
        out_image_exp_params = results.img_hyper_params_list[-9:]
        return results.all_images[-9:], out_image_exp_params[-9:]

    class ExplorerModel:
        """
        This is a base class for models that are executed to create new SBMultiSampleArgs given input of a history of
        user perfeneces and the SBIRenderParams to use. Subclasses should implement the generate_params method
        """

        # def __init__(self):
        #    self.sbi_render_params = sbi_render_params

        def __call__(self, *args, **kwargs):
            return self.generate_params(*args, **kwargs)

        def generate_params(self, render_params: SBIRenderParams, param_history: [SBIHyperParams]) -> SBMultiSampleArgs:
            raise NotImplementedError

    class SimpleExplorerModel(ExplorerModel):
        def __init__(self):
            super().__init__()

        def generate_params(self, render_params: SBIRenderParams, param_history: [SBIHyperParams]) -> SBMultiSampleArgs:
            default_hyper_params = SBIHyperParams(
                prompt="",
                seed=-1,
                negative_prompt="",
                cfg_scale=7,
                subseed=-1,
                subseed_strength=0,
                steps=20,
            )
            base_hyper_params = copy.deepcopy(default_hyper_params)
            base_hyper_params.steps = param_history[-1][0].steps
            base_hyper_params.cfg_scale = param_history[-1][0].cfg_scale
            base_hyper_params.negative_prompt = param_history[-1][0].negative_prompt

            def simple_param_gen_func(history: [SBIHyperParams]) -> SBIHyperParams:
                import random
                # check that history is a valid structure of a list of tubples
                if not isinstance(history, list):
                    raise ValueError("history must be a list")
                if not all([isinstance(h, tuple) for h in history]):
                    raise ValueError("history must be a list of tuples")
                if not all([len(h) == 2 for h in history]):
                    raise ValueError("history must be a list of tuples of length 2")

                print("simple_param_gen_func")
                noise = 0.33

                unzip = list(zip(*history))
                params = unzip[0]
                preferences = unzip[1]

                words_accum = [[i[0], 0, .000001] for i in get_prompt_words_and_weights_list(params[0].prompt)]

                for n_param, pref_for_params in zip(params, preferences):
                    words_and_weights = get_prompt_words_and_weights_list(n_param.prompt)
                    for i in range(len(words_and_weights)):
                        if pref_for_params == 1:
                            # upvote
                            words_accum[i][1] += words_and_weights[i][1]
                            words_accum[i][2] += 1
                        elif pref_for_params == 2:
                            # downvote
                            words_accum[i][1] -= words_and_weights[i][1]
                            words_accum[i][2] += 1
                        elif pref_for_params == 3:
                            # select for story board
                            words_accum[i][1] += words_and_weights[i][1]
                            words_accum[i][2] += 1
                        else:
                            # no vote
                            pass

                word_means = [[i[0], i[1] / i[2]] for i in words_accum]

                # get the default params from the first entry in the history
                out_params = copy.deepcopy(base_hyper_params)
                # out_params.prompt = []
                # out_params.seed = []
                for idx in range(9):
                    new_params = copy.deepcopy(base_hyper_params)
                    # re weight the words based on the mean and generate a new prompt
                    prompt = ""
                    for word in word_means:
                        _noise = (random.random() * noise) - (noise / 2.0)
                        prompt += f"({word[0]}:{word[1] + _noise}) "
                    new_params.prompt = prompt.strip()
                    out_params += new_params

                    # out_params.seed.append(-1)  # means create a random seed

                return out_params

            hyper_params = simple_param_gen_func(param_history)
            return SBMultiSampleArgs(render_params, hyper_params)

    def explore(self, render_params, hyper_params_history, explorer_model: ExplorerModel) -> SBMultiSampleArgs:
        """generates a new set of parameters created by the explorer_model passed in and the history of user preferences
        in hyper_params_history, then returns the new parameters as a SBMultiSampleArgs that can be used to render a new
        set of images for the user to explore"""
        result = explorer_model(render_params, hyper_params_history)
        return result

    def record_and_render(self, all_state, idx, vote_category: int, *ui_param_state):
        # TODO: use this in on_generate
        #       Going to need to create a new explorer model that uses the ui_param_state to generate random h_params

        cell_params = all_state["im_explorer_hparams"][idx]
        all_state["history"].append((cell_params, vote_category))
        render_params = self.get_sb_multi_sample_params_from_ui(ui_param_state).render
        render_call_args = self.explore(render_params, all_state["history"], self.SimpleExplorerModel())
        image_results, all_state["im_explorer_hparams"] = self.render_explorer(render_call_args)

        return all_state, *image_results, *[i.__dict__ for i in all_state["im_explorer_hparams"]]

    def on_upvote(self, idx, all_state, *ui_param_state):
        return self.record_and_render(all_state, idx, 1, *ui_param_state)

    def on_downvote(self, idx, all_state, *ui_param_state):
        return self.record_and_render(all_state, idx, 2, *ui_param_state)

    def on_promote(self, idx, all_state, *args):
        # general pattern:
        # 1. update the params history
        # 2. move the image to the correct position in the storyboard
        # 3. update the storyboard image params
        # 4. generate new params for the image explorer
        # 5. render the new images
        # 6. update the image explorer text
        # 7. return the new params history, new storyboard images, new image explorer images, new image explorer params, new image explorer text

        if "arrange the variables we will be working with":
            ui_state_comps = args[3:]
            story_board_images: [] = list(args[:3])
            cell_params = all_state["im_explorer_hparams"][idx]
            render_params = self.get_sb_multi_sample_params_from_ui(ui_state_comps).render
            sb_idx: int = len(all_state["story_board"])

        if "set simple params":
            all_state["history"].append((cell_params, 4))
            all_state["story_board"].append(cell_params)

        if "handle the storyboard first":
            # render the new storyboard image
            call_args = self.get_sb_multi_sample_params_from_ui(ui_state_comps)
            call_args.hyper = cell_params
            call_args.render.batch_size = 1
            # re-render the storyboard image
            sb_result = self.storyboard(call_args, 0)
            # update story_board_images
            story_board_images[sb_idx] = sb_result.all_images[0]

            # get new SBMultiSampleArgs for the image explorer
            # render_call_args = self.explore(render_params, all_state["history"], self.SimpleExplorerModel())

        if "handle the regeneration of the explorer second":
            tmp = list(self.on_generate(all_state, *ui_state_comps))
            all_state, tmp = tmp[0], tmp[1:]
            exp_images, tmp = tmp[:9], tmp[9:]
            exp_texts, _ = tmp[:9], tmp[9:]

        return all_state, *story_board_images, *exp_images, *exp_texts

    def get_sb_multi_sample_params_from_ui(self, ui_param_state:List[gradio.components.Component]) -> SBMultiSampleArgs:
        """takes the ui state and returns a SBMultiSampleArgs object that can be used to render images"""
        sb_image = SBMultiSampleArgs(
            hyper=[SBIHyperParams(
                prompt=ui_param_state[0],
                negative_prompt=ui_param_state[1],
                steps=ui_param_state[2],
                seed=ui_param_state[10],
                subseed=-1,
                subseed_strength=0,
                cfg_scale=ui_param_state[13],

            )],
            render=SBIRenderParams(
                width=ui_param_state[4],
                height=ui_param_state[5],
                batch_size=9,
                tiling=ui_param_state[7],
                batch_count=1,
                restore_faces=ui_param_state[6],
                sampler_index=ui_param_state[3],
            )
        )
        return sb_image

    def on_generate(self, all_state, *ui_param_state):
        """
        generates the first series of images for the user to explore, clears the history, and returns the new params
        :param all_state: the state of the app
        :param ui_param_state: the state of the ui
        :return: the new state of the app, the new storyboard images, the new image explorer images, the new image explorer text
        """

        def get_random_params_and_images(base_params: SBMultiSampleArgs):
            """
            generates a set of random parameters and renders the images for the image explorer
            """
            print("get_random_params_and_images")
            out_sb_image_hyper_params = []

            def random_pompt_word_weights(prompt_to_randomize: str):
                # if the prompt is a list not a string fix it
                if isinstance(prompt_to_randomize, list) and len(prompt_to_randomize) == 1:
                    prompt_to_randomize = prompt_to_randomize[0]
                elif isinstance(prompt_to_randomize, list) and len(prompt_to_randomize) == 0:
                    prompt_to_randomize = ""

                if prompt_to_randomize == "" or prompt_to_randomize is None:
                    prompt_to_randomize = "this is a test prompt that jumped over a lazy dog and then ran away"

                words, weights = zip(*get_prompt_words_and_weights_list(prompt_to_randomize))
                weights = [(random.random() - .5) + w for w in weights]
                prompt_to_randomize = " ".join([f"({w}:{weights[i]})" for i, w in enumerate(words)])

                return prompt_to_randomize

            out_call_args: SBMultiSampleArgs = SBMultiSampleArgs(render=base_params._render, hyper=[])

            for i in range(9):
                tmp: SBIHyperParams = copy.deepcopy(base_params._hyper[0])
                tmp_prompt = random_pompt_word_weights(base_params._hyper[0].prompt)
                tmp.prompt = [tmp_prompt]
                tmp.seed = [modules.processing.get_fixed_seed(-1)]

                out_call_args += tmp
                out_sb_image_hyper_params.append(tmp)

            sb_image_results: SBImageResults = self.storyboard(out_call_args.combined)

            out_images = sb_image_results.all_images

            # remove the image grid from the result if it exists
            if len(out_images) != 9:
                out_images = out_images[1:]

            return out_images, out_sb_image_hyper_params

        sb_msp = self.get_sb_multi_sample_params_from_ui(ui_param_state)

        images_out, params = get_random_params_and_images(sb_msp)
        text_out = self.update_image_exp_text(params)
        all_state["im_explorer_hparams"] = params
        return all_state, *images_out, *text_out, *[None] * 3

    def get_story_squad_ui(self):
        """
        defines the ui for the story squad app, also sets the initial state of the app
        """
        self.all_components["param_inputs"] = {}

        with gr.Blocks() as param_area:
            with gr.Column(variant='panel'):
                self.ordered_list_of_param_inputs.append(gr.Slider(minimum=1, maximum=150, step=1,
                                                                   label="Sampling Steps",
                                                                   value=5))
                self.all_components["param_inputs"]["steps"] = self.ordered_list_of_param_inputs[-1]

                self.ordered_list_of_param_inputs.append(gr.State(9))
                self.all_components["param_inputs"]["sampler_index"] = self.ordered_list_of_param_inputs[-1]
                with gr.Row():
                    self.ordered_list_of_param_inputs.append(gr.Slider(minimum=64, maximum=2048, step=64,
                                                                       label="Width",
                                                                       value=512))
                    self.all_components["param_inputs"]["width"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(gr.Slider(minimum=64, maximum=2048, step=64,
                                                                       label="Height",
                                                                       value=512))
                    self.all_components["param_inputs"]["height"] = self.ordered_list_of_param_inputs[-1]

                with gr.Row():
                    self.ordered_list_of_param_inputs.append(gr.Checkbox(label='Restore faces', value=False,
                                                                         visible=True))
                    self.all_components["param_inputs"]["restore_faces"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(gr.Checkbox(label='Tiling', value=False))
                    self.all_components["param_inputs"]["tiling"] = self.ordered_list_of_param_inputs[-1]

                with gr.Row():
                    self.ordered_list_of_param_inputs.append(gr.Slider(minimum=1, step=1, label='Batch count', value=1))
                    self.all_components["param_inputs"]["batch_count"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(gr.Slider(minimum=1, maximum=20, step=1, value=1))
                    self.all_components["param_inputs"]["batch_size"] = self.ordered_list_of_param_inputs[-1]

                self.ordered_list_of_param_inputs.append(
                    gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0))
                self.all_components["param_inputs"]["cfg_scale"] = self.ordered_list_of_param_inputs[-1]

                # seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox
                self.ordered_list_of_param_inputs.append(self.create_seed_inputs())

                self.all_components["param_inputs"]["seed"], \
                self.all_components["param_inputs"]["reuse_seed"], \
                self.all_components["param_inputs"]["subseed"], \
                self.all_components["param_inputs"]["reuse_subseed"], \
                self.all_components["param_inputs"]["subseed_strength"], \
                self.all_components["param_inputs"]["seed_resize_from_h"], \
                self.all_components["param_inputs"]["seed_resize_from_w"], \
                self.all_components["param_inputs"]["seed_checkbox"] = self.ordered_list_of_param_inputs[-8:]

        with gr.Blocks() as story_squad_interface:
            id_part = "storyboard_call_multi"

            with gr.Row(elem_id="toprow"):
                with gr.Column(scale=6):
                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                self.all_components["param_inputs"]["prompt"] = gr.Textbox(label="Prompt",
                                                                                           elem_id=f"{id_part}_prompt",
                                                                                           show_label=False,
                                                                                           lines=2,
                                                                                           placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)"
                                                                                           )

                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                self.all_components["param_inputs"]["negative_prompt"] = gr.Textbox(
                                    label="Negative prompt",
                                    elem_id=f"{id_part}_neg_prompt",
                                    show_label=False,
                                    lines=2,
                                    placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)"
                                )

                with gr.Column(scale=1, elem_id="roll_col", visible=False):
                    roll = gr.Button(value="R", elem_id="roll", visible=False)
                    paste = gr.Button(value="P", elem_id="paste", visible=False)
                    save_style = gr.Button(value="S", elem_id="style_create", visible=False)
                    prompt_style_apply = gr.Button(value="A", elem_id="style_apply", visible=False)

                    token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
                    token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")

                button_interrogate = None
                button_deepbooru = None

                with gr.Column(scale=1):
                    with gr.Row():
                        self.all_components["submit"] = gr.Button('Generate', elem_id=f"{id_part}_generate",
                                                                  variant='primary')
                    with gr.Row():
                        with gr.Column(scale=1, elem_id="style_pos_col"):
                            prompt_style = gr.State("None")
                        with gr.Column(scale=1, elem_id="style_neg_col"):
                            prompt_style2 = gr.State("None")
                    with gr.Row():
                        # create some empty space padding with gr.HTML
                        gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")
                    with gr.Row():
                        self.all_components["render"] = gr.Button('Render', elem_id=f"{id_part}_render",
                                                                  variant='primary')

            with gr.Column():
                label = make_gr_label("StoryBoard by Story Squad")
                with gr.Row(scale=1):
                    with gr.Column():
                        param_area.render()

                gr.HTML("<hr>")
                gr.update()
                with gr.Row(scale=1, variant="panel", elem_id="changeme"):  # story board
                    with gr.Column():
                        make_gr_label("StoryBoard")
                        with gr.Row(label="StoryBoard", scale=1):
                            image1 = gr.Image(label="Image 1")
                            image2 = gr.Image(label="Image 2")
                            image3 = gr.Image(label="Image 3")
                            # self.all_components["story_board_image1"] = image1
                            # self.all_components["story_board_image2"] = image2
                            # self.all_components["story_board_image3"] = image3
                            self.all_components["story_board_images"] = [image1, image2, image3]

                gr.HTML("<hr>")
                make_gr_label("Parameter Explorer")

                with gr.Row(scale=1) as image_exolorer:
                    with gr.Blocks():
                        for r in range(3):
                            with gr.Row(equal_height=True):
                                for c in range(3):
                                    with gr.Column(equal_width=True):
                                        with gr.Group():
                                            self.create_img_exp_group()

        return story_squad_interface

    def setup_story_board_events(self):
        """
        Setup the events for the story board using self.all_components that was initialized in __init__ which called
        which called this function.
        """
        self.all_components["param_inputs"]["list_for_generate"] = \
            [self.all_components["param_inputs"][k] for k in
             keys_for_ui_in_order]
        self.all_components["render"].click(self.render_storyboard,
                                            inputs=[self.all_state,
                                                    *self.all_components["param_inputs"]["list_for_generate"]],
                                            outputs=[self.all_state, *self.all_components["im_explorer"]["images"]]
                                            )

        self.all_components["submit"].click(self.on_generate,
                                            inputs=[self.all_state,
                                                    *self.all_components["param_inputs"]["list_for_generate"]],
                                            outputs=[self.all_state,
                                                     *self.all_components["im_explorer"]["images"],
                                                     *self.all_components["im_explorer"]["texts"],
                                                     *self.all_components["story_board_images"]
                                                     ]
                                            )

        # create the events for the image explorer, buttons, and text boxes
        cur_img_idx = 0
        while True:
            # cur_img = self.all_state["im_explorer"]["images"][cur_img_idx]
            but_idx_base = cur_img_idx * 3
            but_up = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                      "up" in b.label.lower()][0]
            but_use = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                       "use" in b.label.lower()][0]
            but_down = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                        "down" in b.label.lower()][0]

            but_up.click(self.on_upvote,
                         inputs=[gr.State(cur_img_idx),
                                 self.all_state,
                                 *self.all_components["param_inputs"]["list_for_generate"]],
                         outputs=[self.all_state,
                                  *self.all_components["im_explorer"]["images"],
                                  *self.all_components["im_explorer"]["texts"], ]
                         )
            but_use.click(self.on_promote,
                          inputs=[gr.State(cur_img_idx),
                                  self.all_state,
                                  *self.all_components["story_board_images"],
                                  *self.all_components["param_inputs"]["list_for_generate"],
                                  ],
                          outputs=[self.all_state,
                                   *self.all_components["story_board_images"],
                                   *self.all_components["im_explorer"]["images"],
                                   *self.all_components["im_explorer"]["texts"], ]
                          )

            but_down.click(self.on_downvote,
                           inputs=[gr.State(cur_img_idx),
                                   self.all_state,
                                   *self.all_components["param_inputs"]["list_for_generate"]],
                           outputs=[self.all_state,
                                    *self.all_components["im_explorer"]["images"],
                                    *self.all_components["im_explorer"]["texts"], ]
                           )

            cur_img_idx += 1
            if cur_img_idx >= 9: break

    def create_img_exp_group(self):
        gr_comps = self.all_components
        if "im_explorer" not in gr_comps:
            gr_comps["im_explorer"] = OrderedDict()
            gr_comps["im_explorer"]["images"] = []
            gr_comps["im_explorer"]["buttons"] = []
            gr_comps["im_explorer"]["texts"] = []

        img = gr.Image(value=random_noise_image(), show_label=False,
                       interactive=False)
        gr_comps["im_explorer"]["images"].append(img)

        with gr.Row():
            with gr.Group():
                but = gr.Button(f"Up", label="Up")
                gr_comps["im_explorer"]["buttons"].append(but)

                but = gr.Button(f"Use", label="Use")
                gr_comps["im_explorer"]["buttons"].append(but)

                but = gr.Button(f"Down", label="Down")
                gr_comps["im_explorer"]["buttons"].append(but)

        text = gr.Textbox(show_label=False, max_lines=3, interactive=False)

        gr_comps["im_explorer"]["texts"].append(text)


if __name__ == '__main__':
    import doctest
    import random
    import gradio as gr
    import numpy as np
    from typing import List
    from collections import OrderedDict

    doctest.run_docstring_examples(StorySquad.render_storyboard, globals(), verbose=True)
