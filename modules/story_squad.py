import gradio as gr
import numpy as np
# import random
# import scripts
# import shared as shared
from modules.sd_samplers import samplers

# from modules import storyboard
from typing import List
# from dataclasses import dataclass
import copy
from collections import OrderedDict


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
    prompt = sanitize_prompt(prompt)
    words = prompt.split(" ")
    o = [i.split(":") for i in words]
    o = [(i[0], 1.0 or float(i[1])) for i in o]
    return o


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

        self.create_toprow = create_toprow
        self.setup_progressbar = setup_progressbar
        self.create_seed_inputs = create_seed_inputs
        self.storyboard = storyboard
        self.wrapper_func = wrapper_func
        self.storyboard_params = []

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
        params_to_use = self.simple_param_gen_func(params_history)

        results = [wrapped_func(p_obj, 0) for p_obj in params_to_use]
        image_results = [r[0][0] for r in results]
        text_out = self.update_image_exp_text(params_to_use)

        return params_history, *image_results, *params_to_use, *text_out

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
        print("\non_promote:")
        print(cell_params)
        print(f"history: {params_history}")

        storyboard_images = [*args[:3]]
        ui_param_state = args[3:]

        # append the selected params to the history
        params_history.append((cell_params, 3))
        # move the params to the storyboard
        board_empty_idx = len(self.storyboard_params)
        self.storyboard_params.append(gr.State(cell_params))

        # re-generate the storyboard image based on the new params
        wrapped_func = self.wrapper_func(self.storyboard)
        results = wrapped_func(cell_params, 0)

        # update the storyboard image
        storyboard_images[board_empty_idx] = results[0][0]

        # generate new params and images
        results = self.on_generate(*ui_param_state)
        exp_images = results[0:9]
        _img_exp_sd_args = results[9:18]
        texts = results[18:27]

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
        out_images = []
        wrapped_func = self.wrapper_func(self.storyboard)

        def random_pompt_word_weights(prompt_to_randomize):
            if prompt_to_randomize == "" or prompt_to_randomize is None:
                prompt_to_randomize = "this is a test prompt that jumped over a lazy dog and then ran away"

            words, weights = zip(*get_prompt_words_and_weights_list(prompt_to_randomize))

            prompt_to_randomize = " ".join([f"({w}:{weights[i]})" for i, w in enumerate(words)])

            return prompt_to_randomize

        for i in range(9):
            out_call_args = copy.deepcopy(p_obj)
            out_call_args.prompt = random_pompt_word_weights(p_obj.prompt)
            out_image_explorer_params.append(out_call_args)

        results = []
        for params in out_image_explorer_params:
            result = wrapped_func(params, extra)
            results.append(result)
            out_images.append(result[0][0])
            params.seed = result[1]

        # for i in range (len(out_image_explorer_params)):
        #    out_image_explorer_params[i].seed = results[i][1]["seed"]

        return out_images, out_image_explorer_params

    def get_story_squad_ui(self):

        img_exp_params = []
        storyboard_params = []
        ui_gr_comps = OrderedDict()
        ui_gr_comps["param_inputs"] = OrderedDict()
        ui_gr_comps["image_explorer"] = OrderedDict()
        ui_gr_comps["story_board"] = OrderedDict()

        with gr.Blocks() as param_area:
            with gr.Column(variant='panel'):
                ui_gr_comps["param_inputs"]["steps"] = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps",
                                                                 value=20)

                ui_gr_comps["param_inputs"]["sampler_index"] = gr.Radio(label='Sampling method',
                                                                        elem_id="storyboard_sampling",
                                                                        choices=[x.name for x in samplers],
                                                                        value=samplers[0].name,
                                                                        type="index")

                with gr.Group():
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

            prompt, roll, txt2img_prompt_style, negative_prompt, txt2img_prompt_style2, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, paste, token_counter, token_button = self.create_toprow(
                is_img2img=False)
            ui_gr_comps["param_inputs"]["prompt"] = prompt
            ui_gr_comps["param_inputs"]["negative_prompt"] = negative_prompt
            params_history = gr.State([])
            # minimal_dict = {k:v for k,v in ui_gr_comps["param_inputs"].items() if k in CallArgsAsGradioComponents()._gr_keys}
            # gr_sd_args = CallArgsAsGradioComponents(**minimal_dict)
            # gr_sd_args = CallArgsAsGradioComponents(**ui_gr_comps["param_inputs"])

            # references to the gradio UI classes that are not held in the state classes above
            images_n_img_explorer = []
            buts_in_img_explroer = []
            text_in_img_explorer = []

            # instances of the state classes
            # image_exp_state = gr.State(
            #    ImageExplorerState([ExplorerCellState(gr_sd_args) for _ in range(9)]))
            # history of parameters voted up or down

            # gr_sd_args = CallArgsAsGradioComponents()

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
                            ui_gr_comps["story_board"]["image1"] = image1
                            ui_gr_comps["story_board"]["image2"] = image2
                            ui_gr_comps["story_board"]["image3"] = image3

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
                                            img_exp_sd_args.append(CallArgsAsData())

                img_exp_sd_args = [gr.State(args) for args in img_exp_sd_args]
                self.setup_story_board_events(ui_gr_comps, img_exp_sd_args, submit, params_history)
                # submit.click(
                #    update_image_exp_text,
                #    inputs =iexp_sd_args_state,
                #    outputs=text_in_img_explorer
                # )

        return story_squad_interface

    def setup_story_board_events(self, ui_gr_comps, img_exp_sd_args: List[type(gr.State)], submit,
                                 params_history: List):

        ui_gr_comps["param_inputs"]["list_for_generate"] = [ui_gr_comps["param_inputs"][k] for k in
                                                            CallArgsAsData()._gr_keys]

        submit.click(self.on_generate,
                     inputs=[*ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)]],
                     outputs=[*ui_gr_comps["image_explorer"]["images"],
                              *img_exp_sd_args,
                              *ui_gr_comps["image_explorer"]["texts"]]
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
            but_use.click(self.on_select,
                          inputs=[
                              img_exp_sd_args[cur_img_idx],
                              params_history,
                              *ui_gr_comps["story_board"].values(),
                              *ui_gr_comps["param_inputs"]["list_for_generate"] + [gr.State(0)],
                          ],
                          outputs=[
                              params_history,
                              *ui_gr_comps["image_explorer"]["images"],
                              *img_exp_sd_args,
                              *ui_gr_comps["story_board"].values(),
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
            but = gr.Button(f"Up", label="Up")
            gr_comps["image_explorer"]["buttons"].append(but)

            but = gr.Button(f"Use", label="Use")
            gr_comps["image_explorer"]["buttons"].append(but)

            but = gr.Button(f"Down", label="Down")
            gr_comps["image_explorer"]["buttons"].append(but)

        text = gr.Textbox(show_label=False, max_lines=3)
        gr_comps["image_explorer"]["texts"].append(text)
