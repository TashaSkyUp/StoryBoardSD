print(__name__)
from typing import List
import gradio as gr

MAX_BATCH_SIZE = 9
DEFAULT_HYPER_PARAMS = {
    "prompt": "",
    "negative_prompt": "",
    "steps": 5,
    "seed": 0,
    "subseed": 0,
    "subseed_strength": 0,
    "cfg_scale": 7,
}

if __name__ != "__main__" and __name__ != "story_squad":
    import random
    import doctest
    import numpy as np
    from modules.sd_samplers import samplers
    from typing import List
    import copy
    from collections import OrderedDict
    import modules
    from modules.processing import StableDiffusionProcessingTxt2Img
else:
    print("Running doctests")


class CallArgsAsData:
    def __init__(self, prompt=None,
                 negative_prompt=None, steps=None, sampler_index=None, width=None,
                 height=None, restore_faces=None, tiling=None,
                 batch_count=None,
                 batch_size=None, seed=None, subseed=None, subseed_strength=None, cfg_scale=None, **kwargs):

        self._gr_keys = ["prompt", "negative_prompt", "steps", "sampler_index", "width", "height", "restore_faces",
                         "tiling", "batch_count", "batch_size",
                         "seed", "subseed", "subseed_strength", "cfg_scale"]

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.sampler_index = sampler_index
        self.width = width
        self.height = height
        self.restore_faces = restore_faces
        self.tiling = tiling
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = subseed_strength
        self.cfg_scale = cfg_scale

        self._attributes_as_list = [self.prompt, self.negative_prompt, self.steps, self.sampler_index, self.width,
                                    self.height, self.restore_faces, self.tiling,
                                    self.batch_count, self.batch_size, self.seed, self.subseed,
                                    self.subseed_strength, self.cfg_scale]

    def __str__(self):
        return ",".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_")])

    def __repr__(self):
        return self.__str__()

    def __setattr__(self, key, value):
        if self.__dict__.get("_gr_keys") is not None:
            if self.__dict__.get("attributes_as_list") is not None:
                if key in self._gr_keys:
                    self._attributes_as_list[self._gr_keys.index(key)] = value
        super().__setattr__(key, value)


class CallArgsAsGradioComponents(CallArgsAsData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check if all components are gradio components
        for component in self._attributes_as_list:
            if not isinstance(component, gr.components.Component):
                raise ValueError("All components must be gradio components")


class ExplorerCellState:
    # the state of the cell that is not held in the gradio UI classes
    def __init__(self, m_params: CallArgsAsGradioComponents = None):
        self.m_params: CallArgsAsGradioComponents = m_params
        self.rendering: bool = False
        self.up_count: int
        self.down_count: int
        self.idle_secs: int = 0
        self.passed_to_model_recently: bool = False


class ImageExplorerState:
    # for states the effect the whole explorer
    def __init__(self, cells: List[ExplorerCellState]):
        self.cells = cells


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
    # o = [(i[0], 1.0 or float(i[1])) for i in o]
    out = []
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
            except:
                raise ValueError("Prompt weights must be numbers")

        out.append((word_weight_pair[0], w))
    return out


def random_noise_image():
    return np.random.rand(512, 512, 3)


def make_gr_label(label):
    return gr.HTML(f"<div class = 'flex ont-bold text-2xl items-center justify-center'> {label} </div>")


class StorySquad:
    # from modules.ui import create_toprow, setup_progressbar, create_seed_inputs
    # import ui
    def __init__(self, wrapper_func):
        from modules.ui import create_toprow, setup_progressbar, create_seed_inputs
        from modules.storyboard import storyboard as storyboard

        # self.create_toprow = create_toprow
        self.setup_progressbar = setup_progressbar
        self.create_seed_inputs = create_seed_inputs
        self.storyboard = storyboard
        self.wrapper_func = wrapper_func
        self.storyboard_params = []

    def render_storyboard(self, *args, **kwargs) -> List[object]:
        """
        >>> StorySquad.render_storyboard([CallArgsAsData(prompt= "(dog:1) cat:0",seed=1),CallArgsAsData(prompt= "(dog:1) cat:1",seed=2),CallArgsAsData(prompt= "(dog:0) cat:1",seed=3)],test=True)
        render_storyboard
        ()
        test
        [prompt: (dog:1.0) (cat:0.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 1,subseed: 2,subseed_strength: None,cfg_scale: None,sub_seed_weight: 0.0, prompt: (dog:1.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 1,subseed: 2,subseed_strength: None,cfg_scale: None,sub_seed_weight: 1.0, prompt: (dog:1.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 2,subseed: 3,subseed_strength: None,cfg_scale: None,sub_seed_weight: 0.0, prompt: (dog:0.0) (cat:1.0),negative_prompt: None,steps: None,sampler_index: None,width: None,height: None,restore_faces: None,tiling: None,batch_count: None,batch_size: None,seed: 2,subseed: 3,subseed_strength: None,cfg_scale: None,sub_seed_weight: 1.0]
        """
        print("render_storyboard")
        # storyboard_params = args[:3]
        img_exp_sd_args = args[3:]
        print(img_exp_sd_args)
        storyboard_params = [s.value for s in self.storyboard_params]

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

            # define the two distinct sections of the storyboard
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

        def get_frame_seed_data(board_params, num_frames) -> [()]:  # List[(seed,subseed,weight)]
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
                for i in range(num_frames):
                    sub_seed_weight = i / (num_frames - 1)
                    all_frames.append((seed, subseed, sub_seed_weight))
            return all_frames

        if "test" in kwargs.keys():
            test = kwargs["test"]
        else:
            test = False
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

        # create a base CallArgsAsData object to use as a template
        base_params = storyboard_params[0]

        # turn the list of prompts and seeds into a list of CallArgsAsData using the base_params as a template
        args_for_render = []
        for prompt, seed in zip(prompts, seeds):
            base_copy = copy.deepcopy(base_params)
            base_copy.prompt = prompt
            base_copy.seed = seed[0]
            base_copy.subseed = seed[1]
            base_copy.subseed_strength = seed[2]
            args_for_render.append(base_copy)

        wrapped_func = self.wrapper_func(self.storyboard)
        images_to_save = []
        if not test:
            for args in args_for_render:
                results = wrapped_func(args, 0)
                images_to_save.append(results[0][0])

        return args_for_render

    def update_image_exp_text(self, img_exp_sd_args):
        print("update_image_exp_text")
        print(img_exp_sd_args)
        o = [str(i) for i in img_exp_sd_args]
        return o

    def event_update_image_explorer(self, explorer_state):
        print("event_update_image_explorer")
        print(explorer_state)
        for r in range(3):
            for c in range(3):
                explorer_state["images"][r][c].value = random_noise_image()
                explorer_state["buttons"][r][c].label = f"favorite: {r},{c}"

    def simulate_model_response(self, new_params, weight, explorer_state: ImageExplorerState, params_history_in):
        # this simulates what the model does to the state before passing it back
        # randomize the parameters that the model will use

        print("simulate_model_response")

        exp_state_out = explorer_state
        p_hist_out = params_history_in
        p_hist_out.append(new_params)

        if not new_params: new_params = "RF"

        images_out = []

        for cell in exp_state_out.cells:
            cell.m_params.seed += 1
            images_out.append(random_noise_image())

        return exp_state_out, images_out, p_hist_out

    def simple_param_gen_func(self, param_history):
        import random
        print("simple_param_gen_func")
        noise = 0.33

        unzip = list(zip(*param_history))
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

        out_params = []
        for ret_param in range(9):
            # re weight the words based on the mean and generate a new prompt
            prompt = ""
            for word in word_means:
                _noise = (random.random() * noise) - (noise / 2.0)
                prompt += f"({word[0]}:{word[1] + _noise}) "

            cpy = copy.deepcopy(params[0])
            cpy.prompt = prompt.strip()

            # this is to assure that the user is seeing a variety of seeds / content
            cpy.seed = -1  # means create a random seed
            out_params.append(cpy)

        return out_params

    def on_upvote(self, cell_params, params_history, *ui_param_state):
        print("on_upvote")
        print(cell_params)
        wrapped_func = self.wrapper_func(self.storyboard)
        params_history.append((cell_params, 1))
        params_for_new_img_exp = self.simple_param_gen_func(params_history)

        image_results = []
        for params in params_for_new_img_exp:
            result = wrapped_func(params, 0)
            image_results.append(result[0][0])
            params.seed = result[1]

        text_out = self.update_image_exp_text(params_for_new_img_exp)
        return params_history, *image_results, *params_for_new_img_exp, *text_out

    def on_downvote(self, cell_params, params_history, *ui_param_state):
        print("on_upvote")
        print(cell_params)
        wrapped_func = self.wrapper_func(self.storyboard)
        params_history.append((cell_params, 2))
        params_to_use = self.simple_param_gen_func(params_history)

        results = [wrapped_func(p_obj, 0) for p_obj in params_to_use]
        image_results = [r[0][0] for r in results]
        text_out = self.update_image_exp_text(params_to_use)

        return params_history, *image_results, *params_to_use, *text_out

    def on_promote(self, cell_params, params_history, *args):
        retool = True

        print("\non_promote:")
        print(cell_params)
        print(f"history: {params_history}")

        storyboard_images = [*args[:3]]
        ui_param_state = args[3:]

        # append the selected params to the history
        params_history.append((cell_params, 3))

        # find an empty slot in the storyboard
        board_empty_idx = 0
        for i in range(len(self.storyboard_params)):
            if self.storyboard_params[i].value.prompt is None:
                board_empty_idx = i
                break
        # move the params to the storyboard

        self.storyboard_params[board_empty_idx] = gr.State(cell_params)
        # re-generate the storyboard image based on the new params

        wrapped_func = self.wrapper_func(self.storyboard)
        results = wrapped_func(cell_params, 0)

        # update the storyboard image
        storyboard_images[board_empty_idx] = results[0][0]

        if not retool:
            # generate new params and images
            results = self.on_generate(*ui_param_state)
            exp_images = results[0:9]
            _img_exp_sd_args = results[9:18]
            texts = results[18:27]
        else:
            # generate noise for images
            exp_images = [random_noise_image() for _ in range(9)]
            # generate new empty params
            _img_exp_sd_args = [CallArgsAsData() for _ in range(9)]
            # set param_history to empty
            params_history = []
            # clear the texts
            texts = [""] * 9

        return params_history, *exp_images, *_img_exp_sd_args, *storyboard_images, *texts

    def on_generate(self, *ui_param_state):
        param_list_len = 14
        print("on_generate")

        prompt, negative_prompt, steps, sampler_index, width, height, restore_faces, tiling, \
        batch_count, batch_size, seed, subseed, subseed_strength, cfg_scale = \
            ui_param_state[:param_list_len]

        extra = ui_param_state[param_list_len:]

        p_obj = CallArgsAsData(prompt, negative_prompt, steps, sampler_index, width, height, restore_faces,
                               tiling, batch_count, batch_size,
                               seed, subseed, subseed_strength, cfg_scale)

        images_out, params = self.get_random_params_and_images(p_obj, *extra)
        text_out = self.update_image_exp_text(params)
        return *images_out, *params, *text_out, []

    def get_random_params_and_images(self, p_obj, *extra):
        print("get_random_params_and_images")
        out_image_explorer_params = []

        wrapped_func = self.wrapper_func(self.storyboard)

        def random_pompt_word_weights(prompt_to_randomize):
            if prompt_to_randomize == "" or prompt_to_randomize is None:
                prompt_to_randomize = "this is a test prompt that jumped over a lazy dog and then ran away"

            words, weights = zip(*get_prompt_words_and_weights_list(prompt_to_randomize))
            weights = [(random.random() - .5) + w for w in weights]
            prompt_to_randomize = " ".join([f"({w}:{weights[i]})" for i, w in enumerate(words)])

            return prompt_to_randomize

        out_call_args = copy.deepcopy(p_obj)
        out_call_args.prompt = []
        out_call_args.batch_size = 9

        out_call_args.seed = []
        for i in range(9):
            tmp = copy.deepcopy(p_obj)
            tmp_prompt = random_pompt_word_weights(p_obj.prompt)
            out_call_args.prompt.append(tmp_prompt)
            out_call_args.seed.append(modules.processing.get_fixed_seed(-1))
            tmp.prompt = tmp_prompt
            out_image_explorer_params.append(tmp)

        modules.processing.fix_seed(out_call_args)
        result = wrapped_func(out_call_args, extra)

        out_images = result[0]

        # remove the image grid from the result if it exists
        if len(out_images) != 9:
            out_images = out_images[1:]

        return out_images, out_image_explorer_params

    def get_story_squad_ui(self):

        img_exp_params = []
        storyboard_params = []
        ui_gr_comps = OrderedDict()
        ui_gr_comps["param_inputs"] = OrderedDict()
        ui_gr_comps["image_explorer"] = OrderedDict()
        ui_gr_comps["story_board"] = [gr.State(CallArgsAsData()) for _ in range(3)]
        self.storyboard_params = ui_gr_comps["story_board"]

        with gr.Blocks() as param_area:
            with gr.Column(variant='panel'):
                ui_gr_comps["param_inputs"]["steps"] = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps",
                                                                 value=20)

                ui_gr_comps["param_inputs"]["sampler_index"] = gr.State(9)
                with gr.Row():
                    ui_gr_comps["param_inputs"]["width"] = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                                     value=512)
                    ui_gr_comps["param_inputs"]["height"] = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                                      value=512)

                with gr.Row():
                    ui_gr_comps["param_inputs"]["restore_faces"] = gr.Checkbox(label='Restore faces', value=False,
                                                                               visible=True)
                    ui_gr_comps["param_inputs"]["tiling"] = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    ui_gr_comps["param_inputs"]["batch_count"] = gr.Slider(minimum=1, step=1, label='Batch count',
                                                                           value=1)
                    ui_gr_comps["param_inputs"]["batch_size"] = gr.Slider(minimum=1, maximum=20, step=1, value=1)

                ui_gr_comps["param_inputs"]["cfg_scale"] = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                                     label='CFG Scale', value=7.0)
                # seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox
                ui_gr_comps["param_inputs"]["seed"], \
                ui_gr_comps["param_inputs"]["reuse_seed"], \
                ui_gr_comps["param_inputs"]["subseed"], \
                ui_gr_comps["param_inputs"]["reuse_subseed"], \
                ui_gr_comps["param_inputs"]["subseed_strength"], \
                ui_gr_comps["param_inputs"]["seed_resize_from_h"], \
                ui_gr_comps["param_inputs"]["seed_resize_from_w"], \
                ui_gr_comps["param_inputs"]["seed_checkbox"] = self.create_seed_inputs()

                # import modules.scripts

                # do we even want this? is it easier to remove it than to fix it?
                # with gr.Group():
                # ui_gr_comps["param_inputs"]["custom_inputs"] = modules.scripts.scripts_storyboard.setup_ui(is_img2img=False)  #

        with gr.Blocks() as story_squad_interface:

            id_part = "storyboard"
            params_history = gr.State([])

            with gr.Row(elem_id="toprow"):
                with gr.Column(scale=6):
                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                ui_gr_comps["param_inputs"]["prompt"] = gr.Textbox(label="Prompt",
                                                                                   elem_id=f"{id_part}_prompt",
                                                                                   show_label=False,
                                                                                   lines=2,
                                                                                   placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)"
                                                                                   )

                    with gr.Row():
                        with gr.Column(scale=80):
                            with gr.Row():
                                ui_gr_comps["param_inputs"]["negative_prompt"] = gr.Textbox(label="Negative prompt",
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
                        submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                    with gr.Row():
                        with gr.Column(scale=1, elem_id="style_pos_col"):
                            prompt_style = gr.State("None")
                        with gr.Column(scale=1, elem_id="style_neg_col"):
                            prompt_style2 = gr.State("None")
                    with gr.Row():
                        # create some empty space padding with gr.HTML
                        gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")
                    with gr.Row():
                        render = gr.Button('Render', elem_id=f"{id_part}_render", variant='primary')
                        render.click(
                            self.render_storyboard,
                            inputs=[
                                # *ui_gr_comps["story_board"],
                                # *ui_gr_comps["param_inputs"].values()
                            ],
                            outputs=None
                        )

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
                            ui_gr_comps["story_board_image1"] = image1
                            ui_gr_comps["story_board_image2"] = image2
                            ui_gr_comps["story_board_image3"] = image3

                gr.HTML("<hr>")
                make_gr_label("Parameter Explorer")

                with gr.Row(scale=1) as image_exolorer:
                    with gr.Blocks():
                        for r in range(3):
                            with gr.Row(equal_height=True):
                                for c in range(3):
                                    with gr.Column(equal_width=True):
                                        with gr.Group():
                                            self.create_img_exp_group(ui_gr_comps)
                                            img_exp_params.append(CallArgsAsData())

                img_exp_params = [gr.State(args) for args in img_exp_params]

                self.setup_story_board_events(ui_gr_comps, img_exp_params, submit, params_history)
                # submit.click(
                #    update_image_exp_text,
                #    inputs =iexp_sd_args_state,
                #    outputs=text_in_img_explorer
                # )

        return story_squad_interface

    def setup_story_board_events(self, ui_gr_comps, img_exp_sd_args: List[type(gr.State)], submit,
                                 params_history: List):

        # ui_gr_comps["story_board"]["image1"].
        ui_gr_comps["param_inputs"]["list_for_generate"] = [ui_gr_comps["param_inputs"][k] for k in
                                                            CallArgsAsData()._gr_keys]

        submit.click(self.on_generate,
                     inputs=[*ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)]],
                     outputs=[*ui_gr_comps["image_explorer"]["images"],
                              *img_exp_sd_args,
                              *ui_gr_comps["image_explorer"]["texts"],
                              params_history
                              ]
                     )
        cur_img_idx = 0
        while True:
            # cur_img = ui_gr_comps["image_explorer"]["images"][cur_img_idx]
            but_idx_base = cur_img_idx * 3
            but_up = [b for b in ui_gr_comps["image_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                      "up" in b.label.lower()][0]
            but_use = [b for b in ui_gr_comps["image_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                       "use" in b.label.lower()][0]
            but_down = [b for b in ui_gr_comps["image_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                        "down" in b.label.lower()][0]

            but_up.click(self.on_upvote,
                         inputs=[
                             img_exp_sd_args[cur_img_idx],
                             params_history,
                             *ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)],
                         ],
                         outputs=[
                             params_history,
                             *ui_gr_comps["image_explorer"]["images"],
                             *img_exp_sd_args,
                             *ui_gr_comps["image_explorer"]["texts"]
                         ]

                         )
            but_use.click(self.on_promote,
                          inputs=[
                              img_exp_sd_args[cur_img_idx],
                              params_history,
                              ui_gr_comps["story_board_image1"],
                              ui_gr_comps["story_board_image2"],
                              ui_gr_comps["story_board_image3"],
                              *ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)],
                          ],
                          outputs=[
                              params_history,
                              *ui_gr_comps["image_explorer"]["images"],
                              *img_exp_sd_args,
                              ui_gr_comps["story_board_image1"],
                              ui_gr_comps["story_board_image2"],
                              ui_gr_comps["story_board_image3"],
                              *ui_gr_comps["image_explorer"]["texts"]
                          ]
                          )
            but_down.click(self.on_downvote,
                           inputs=[
                               img_exp_sd_args[cur_img_idx],
                               params_history,
                               *ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)],
                           ],
                           outputs=[
                               params_history,
                               *ui_gr_comps["image_explorer"]["images"],
                               *img_exp_sd_args,
                               *ui_gr_comps["image_explorer"]["texts"]
                           ]
                           )

            cur_img_idx += 1
            if cur_img_idx >= 9: break

    def create_img_exp_group(self, gr_comps):
        if not gr_comps["image_explorer"]:
            gr_comps["image_explorer"] = OrderedDict()
            gr_comps["image_explorer"]["images"] = []
            gr_comps["image_explorer"]["buttons"] = []
            gr_comps["image_explorer"]["texts"] = []

        img = gr.Image(value=random_noise_image(), show_label=False,
                       interactive=False)
        gr_comps["image_explorer"]["images"].append(img)

        with gr.Row():
            with gr.Group():
                but = gr.Button(f"Up", label="Up")
                gr_comps["image_explorer"]["buttons"].append(but)

                but = gr.Button(f"Use", label="Use")
                gr_comps["image_explorer"]["buttons"].append(but)

                but = gr.Button(f"Down", label="Down")
                gr_comps["image_explorer"]["buttons"].append(but)

        text = gr.Textbox(show_label=False, max_lines=3, interactive=False)

        gr_comps["image_explorer"]["texts"].append(text)


if __name__ == '__main__':
    import doctest
    import random
    import gradio as gr
    import numpy as np
    from typing import List
    from collections import OrderedDict

    doctest.run_docstring_examples(StorySquad.render_storyboard, globals(), verbose=True)
