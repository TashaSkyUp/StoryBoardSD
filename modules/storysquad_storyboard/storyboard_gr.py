from modules.storysquad_storyboard.branched_renderer import compose_storyboard_render_dev
MAX_SB_SIZE = 3
print(__name__)
from modules.storysquad_storyboard.sb_rendering import *
from modules.storysquad_storyboard.testing import get_test_storyboard
import gradio.components
import json
from dataclasses import dataclass
from typing import List

keys_for_ui_in_order = ["prompt", "negative_prompt", "steps", "sampler_index", "width", "height", "restore_faces",
                         "tiling", "batch_count", "batch_size",
                         "seed", "subseed", "subseed_strength", "cfg_scale"]


# TODO: consider adding a feature to render more results than the Image explorer can show at one time
MAX_IEXP_SIZE = 9


@dataclass
class BenchMarkSettings:
    steps_to_test = [3, 4, 5, 6]
    fps_targets = [4, 8, 24, 30]
    num_frames_per_sctn = 5
    stop_after_mins = 10


if __name__ != "__main__" and __name__ != "storyboard_gr":
    from modules.storysquad_storyboard.storyboard import DEFAULT_HYPER_PARAMS_DICT
    from modules.storysquad_storyboard.sb_sd_render import *
    import modules.storysquad_storyboard.env as sb_env

    import random
    import doctest
    import numpy as np
    from typing import List
    import copy
    from collections import OrderedDict
    import modules
    from modules.processing import StableDiffusionProcessingTxt2Img

    DEV_MODE = sb_env.STORYBOARD_DEV_MODE
    if DEV_MODE:
        ONLY_USE_NOUNS = False
    else:
        ONLY_USE_NOUNS = False
else:
    # TODO: figure out how to load the correct modules when running this file directly for doctests
    import copy
    print("Running doctests")


def random_noise_image():
    return np.random.rand(512, 512, 3)


def make_gr_label(label):
    import gradio as gr
    return gr.HTML(f"<div class = 'flex ont-bold text-2xl items-center justify-center'> {label} </div>")


class ExplorerModel:
    """
    This is a base class for models that are executed to create new SBMultiSampleArgs given input of a history of
    user perfeneces and the SBIRenderParams to use. Subclasses should implement the generate_params method
    """

    def __call__(self, *args, **kwargs):
        return self.generate_params(*args, **kwargs)

    def generate_params(self, render_params: SBIRenderParams, param_history: [SBIHyperParams]) -> SBMultiSampleArgs:
        raise NotImplementedError

class SimpleExplorerModel(ExplorerModel):
    """

    """
    def __init__(self):
        super().__init__()

    def generate_params(self, render_params: SBIRenderParams, param_history: [SBIHyperParams]) -> SBMultiSampleArgs:

        base_hyper_params = copy.deepcopy(param_history[-1][0])
        base_hyper_params.seed=[-1]
        base_hyper_params.subseed=[-1]
        base_hyper_params.subseed_strength=[0]

        def simple_param_gen_func(history: [SBIHyperParams], num:int) -> SBIHyperParams:

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

            out_params = copy.deepcopy(base_hyper_params)
            # out_params.prompt = []
            # out_params.seed = []
            for idx in range(num):
                new_params = copy.deepcopy(base_hyper_params)
                # re weight the words based on the mean and generate a new prompt
                prompt = ""
                for word in word_means:
                    _noise = (random.random() * noise) - (noise / 2.0)
                    prompt += f"({word[0]}:{word[1] + _noise}) "
                new_params.prompt = prompt.strip()
                out_params += new_params

            return out_params

        hyper_params = simple_param_gen_func(param_history,num=9)[1:]
        return SBMultiSampleArgs(render_params, hyper_params)


def get_files_at_path(path=None) -> List[str]:
    """
    >>> get_files_at_path(path="/home/ubuntu/StoryBoardSD_dev/SB_RENDERS")
    []
    """
    root_files = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f))]
    root_files.sort(key=os.path.getmtime, reverse=True)
    return root_files





class StoryBoardGradio:
    def __init__(self):
        class comp_helper:
            story_squad_interface = None
            render_button = None
            generate_button = None
            robot_dreams = None
            files_interface = None

        self.comp_helper = comp_helper

        self.DefaultRender = DefaultRender()

        # import some functionality from the provided webui
        from modules.ui import setup_progressbar, create_seed_inputs

        # import custom sampler caller and assign it to be easy to access
        #from .sb_sd_render import storyboard_call_endpoint as storyboardtmp
        if STORYBOARD_API_ROLE == "ui_only":
            from .sb_sd_render import storyboard_call_endpoint as storyboardtmp
            from .sb_sd_render import storyboard_call_endpoint_split_batch as storyboardtmp2
        elif STORYBOARD_API_ROLE == "app":
            from .sb_sd_render import storyboard_call_multi as storyboardtmp
            from .sb_sd_render import storyboard_call_multi as storyboardtmp2
        elif STORYBOARD_API_ROLE == "sd_server":
            pass
        self.sd_sb_renderer_batched = storyboardtmp
        self.sd_sb_renderer_s_batch = storyboardtmp2

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
            self.setup_story_board_events()
    def drag_image_in(self, idx, img, all_state, ui_prompt, negative_prompt):
        """
        This function drags an image into a Gradio application and updates the state of the storyboard based on the image's parameters.

        Args:
        idx (int): The index at which the image will be inserted into the storyboard.
        img (PIL.Image): The image to be inserted into the storyboard.
        all_state (dict): The current state of the storyboard.
        prompt (str): The current prompt text.
        negative_prompt (str): The current negative prompt text.

        Returns:
        tuple: A tuple containing the updated state of the storyboard, the inserted image, the updated prompt text, and the updated negative prompt text.

        Raises:
        Exception: If the index is out of range for the storyboard.

        ...
        Example:
        >>> from unittest.mock import MagicMock # doctest: +NORMALIZE_WHITESPACE
        >>> class MockStoryBoardGradio(StoryBoardGradio):
        ...     def __init__(self):
        ...         self.drag_image_in = lambda a, b, c, d, e: StoryBoardGradio.drag_image_in(self, a, b, c, d, e)
        ...         StoryBoardGradio.__init__ = MagicMock(return_value=None)
        ...         StoryBoardGradio.__init__(self)
        >>> sb = MockStoryBoardGradio()
        >>> idx = 0
        >>> from PIL import Image
        >>> img = Image.new('RGB', (50, 50), color='white')  # Creating a blank white image
        >>> img.info=test_img_info
        >>> all_state = {"story_board": [SBIHyperParams(), SBIHyperParams(), SBIHyperParams()]}
        >>> prompt = "Once upon a time..."
        >>> neg_prompt = "Do not use any magic."
        >>> upd_state, upd_img, upd_prompt, upd_neg_prompt = sb.drag_image_in(idx, img, all_state, prompt, neg_prompt)
        >>> upd_state["story_board"][idx]._prompt
        ['(A:1.0389916936335184) (robot:0.9737987777913438)']
        >>> len(upd_state["story_board"]) == 3
        True
        >>> upd_state["story_board"][idx]._prompt
        ['(A:1.0389916936335184) (robot:0.9737987777913438)']
        >>> upd_prompt
        'A robot'
        >>> upd_neg_prompt
        'confusing amateur strange odd ugly mutant disfigured blurry random meme simple bland poor composition toy cartoon photograph'
        """


        import gradio as gr
        new_prompt = None
        if img and "parameters" in img.info.keys():
            #print(img.info)
            params = img.info["parameters"]
            params_split = params.splitlines()
            new_prompt = params_split[0]

            if len(params_split) == 3:
                negative_prompt = params_split[1].split("Negative prompt: ")[1]
            else:
                negative_prompt = negative_prompt

            part3=params_split[-1]
            bracket_idx = part3.find("\"[")
            if bracket_idx > 0:
                brkt2 = part3.find("]\"")
                brktd = part3[bracket_idx:brkt2+2]
                vals = eval(eval(brktd))
                part3 = part3.replace(brktd, f"[{vals[0]}]")
            ps = part3.split(",")
            params={}
            for i in ps:
                try:
                    k, v = i.split(":")
                    params[k.strip()] = v
                except:
                    raise Exception(f"Could not parse {i} in {part3}")

            sbih = SBIHyperParams()
            sbih.prompt = new_prompt
            sbih.negative_prompt = negative_prompt
            sbih.steps = int(params["Steps"])
            sbih.seed = int(params["Seed"])
            if "Variation seed" in params.keys():
                sbih.subseed = int(params["Variation seed"])
                subss = eval(params["Variation seed strength"])[0]
                sbih.subseed_strength = float(subss)
            else:
                sbih.subseed = -1
                sbih.subseed_strength = 0.0

            sbih.cfg_scale = float(params["CFG scale"])

            if idx < len(all_state["story_board"]):
                all_state["story_board"][idx] = sbih
            elif all_state["story_board"]==[]:
                all_state["story_board"]=[sbih]+[None] * (NUM_SB_IMAGES-1)
            else:
                raise Exception(f"index {idx} is out of range for story board")

        if ui_prompt==new_prompt or new_prompt is None:
            out_prompt =gr.Textbox.update()
        else:
            #out_prompt = " ".join([i.split(":")[0][1:] for i in new_prompt.split(" ")])
            from modules.storysquad_storyboard.storyboard import get_prompt_words_list
            out_prompt = get_prompt_words_list(new_prompt,punc=True)
            out_prompt =" ".join(out_prompt)
        return all_state,img,out_prompt,negative_prompt

    def reset_story_board(self, state):
        import gradio as gr
        from modules.storysquad_storyboard.storyboard import get_prompt_words_list
        # TODO: make sure this resets the render button and the generate button and the image explorerer area
        all_state = state[self.all_state]

        num_slots_used =  sum(1 for item in all_state["story_board"] if item is not None)
        if num_slots_used > 0:
            new_prompt = state[self.comp_helper.prompt]
            if len(new_prompt) > 0:
                new_prompt = get_prompt_words_list(new_prompt,punc=True)
            new_prompt =" ".join(new_prompt)
            old_prompt = all_state["story_board"][0].prompt
            old_prompt = get_prompt_words_list(old_prompt,punc=True)
            old_prompt =" ".join(old_prompt)

            if new_prompt==old_prompt:
                return {self.all_state: all_state}

        all_state["story_board"] = [None]*NUM_SB_IMAGES
        all_state["history"] = []
        all_state["im_explorer_hparams"] = []
        sb_imgs = []

        sb_img_dict = {k:random_noise_image() for k
                       in self.all_components["story_board_images"]
                       +self.all_components["im_explorer"]["images"]}

        sb_txt_dict = {k:"" for k in self.all_components["im_explorer"]["texts"]}

        if sb_env.STORYBOARD_PRODUCT=="clash":
            return {
                self.all_state:all_state,
                **sb_img_dict,
                **sb_txt_dict,
                self.comp_helper.robot_dreams:gr.Column.update(visible=True),
                self.comp_helper.render_button:gr.update(visible=False),
                self.comp_helper.generate_button:gr.update(visible=True),
            }
        elif sb_env.STORYBOARD_PRODUCT=="market":
            return {
                self.all_state:all_state,
                **sb_img_dict,
                **sb_txt_dict,
                self.comp_helper.robot_dreams:gr.Column.update(visible=True),
                self.comp_helper.render_button:gr.update(visible=False),
                self.comp_helper.generate_button:gr.update(visible=True),
            }
        if sb_env.STORYBOARD_PRODUCT=="expert":
            return {
                self.all_state:all_state,
                **sb_img_dict,
                **sb_txt_dict,
                self.comp_helper.robot_dreams:gr.Column.update(visible=True),
                self.comp_helper.render_button:gr.update(visible=True),
                self.comp_helper.generate_button:gr.update(visible=True),
            }

    def render_storyboard_benchmark(self, *args):

        # TODO: this currently does not save the benchmark renders to the correct folder

        import os

        all_state = args[0]
        ui_params = list(args[1:])

        # get the parameters for the benchmark from the class
        benchmark_params = BenchMarkSettings()
        fps_targets = benchmark_params.fps_targets
        num_frames_per_sctn = benchmark_params.num_frames_per_sctn
        steps_to_test = benchmark_params.steps_to_test
        stop_after_mins = benchmark_params.stop_after_mins

        out_args=[]

        # iterate through each combination of steps and fps, create new all_state and ui_params for each
        for steps in steps_to_test:
            for fps in fps_targets:
                # create a new all_state and ui_params
                new_all_state = copy.deepcopy(all_state)
                new_ui_params = copy.deepcopy(ui_params)

                new_ui_params[2] = steps

                out_args.append({"steps":steps,"fps":fps,"all_state":all_state, "params":new_ui_params})


        # now create the movies
        # get BENCHMARKS folder from the environment
        benchmark_folder = os.environ.get("STORYBOARD_BENCHMARKS_PATH")
        #verify that the variable is set
        if benchmark_folder is None:
            print("STORYBOARD_BENCHMARKS_PATH environment variable is not set")
            return

        # check the format of the string
        if benchmark_folder[-1] != os.sep:
            benchmark_folder = benchmark_folder + os.sep

        # verify the folder exists
        if benchmark_folder == None:
            print("STORYBOARD_BENCHMARKS_PATH not set")
            return

        # create the folder if it does not exist
        if not os.path.exists(benchmark_folder):
            os.makedirs(benchmark_folder)

        mp4_at_steps={str(i):"" for i in steps_to_test}
        for combo in out_args:
            steps, fps, all_state, new_ui_params = combo["steps"], combo["fps"], combo["all_state"], combo["params"]
            if mp4_at_steps[str(steps)] == "":
                _,mp4_at_steps[str(steps)] = self.on_render_storyboard(stop_after_mins * 60, all_state, *new_ui_params)

            mp4_filename = f"steps_{steps}_fps_{fps}"
            #benchmark_folder, benchmark_folder,mp4_filename
            # copy the mp4 to the benchmark folder
            mp4_path = os.path.join(benchmark_folder, mp4_filename)
            # use os.system to copy the file
            os.system(f"mv {mp4_at_steps[str(steps)]} {mp4_path}.mp4")
        return [all_state,""]

    def compose_storyboard_render_new(self, all_state, early_stop, storyboard_params, test,
                                  test_render, ui_params):
        pass
    def on_render_storyboard_dev(self, early_stop: float = 3, *args):
        """
        """
        print("on_render_storyboard")

        all_state = args[0]
        ui_params = args[1:]

        storyboard_params = all_state["story_board"]
        test_render = False
        if all(map(lambda x: x is None, storyboard_params) or len(storyboard_params) == 0):
            """Test that is ran when the storyboard_params are not set but the user presses the render button"""
            test = True
            test_render = True
            storyboard_params = get_test_storyboard()
            render_settings = DefaultRender()


        else:
            test = False
            storyboard_params = all_state["story_board"]
            ui_params = args[1:]


        complete_mp4_f_path = compose_storyboard_render_dev(self.DefaultRender,
                                                            storyboard_params,
                                                            ui_params,
                                                            self.sd_sb_renderer_batched,
                                                            test,
                                                            early_stop,
                                                            )
        return [all_state,complete_mp4_f_path]

    def on_render_storyboard(self, *args):
        """
        """
        print("on_render_storyboard")

        all_state = args[1]
        ui_params = args[2:]
        early_stop = args[0]

        storyboard_params = all_state["story_board"]
        test_render = False
        if all(map(lambda x: x is None, storyboard_params) or len(storyboard_params) == 0):
            """Test that is ran when the storyboard_params are not set but the user presses the render button"""
            test = True
            test_render = True
            storyboard_params = get_test_storyboard()

        else:
            test = False
            storyboard_params = all_state["story_board"]
            #ui_params = args[1:]


        all_state, complete_mp4_f_path = compose_storyboard_render(self.DefaultRender,
                                                                   all_state,
                                                                   early_stop,
                                                                   storyboard_params,
                                                                   test,
                                                                   test_render,
                                                                   ui_params,
                                                                   SBIMA_render_func=self.sd_sb_renderer_batched,
                                                                   base_SBIMulti=self.get_sb_multi_sample_params_from_ui(ui_params)
                                                                   )
        return [all_state,complete_mp4_f_path]

    def update_image_exp_text(self, h_params:List[SBIHyperParams]):
        # TODO: this needs redone, currently its behavour is not consistent or defined purposfully
        print("update_image_exp_text")
        print(h_params)
        o = [str(i) for i in h_params]
        return o

    async def render_explorer(self, params: SBMultiSampleArgs):
        # calls self.storyboard with the params
        # returns a list of images, and a list of params
        # this is the function that is called by the image explorer
        # it is called with a SBMultiSampleArgs object

        results: SBImageResults = await self.sd_sb_renderer_batched(params, 0)
        out_image_exp_params = results.img_hyper_params_list[-9:]
        return results.all_images[-9:], out_image_exp_params[-9:]

    def explore(self, render_params, hyper_params_history, explorer_model: ExplorerModel) -> SBMultiSampleArgs:
        """generates a new set of parameters created by the explorer_model passed in and the history of user preferences
        in hyper_params_history, then returns the new parameters as a SBMultiSampleArgs that can be used to render a new
        set of images for the user to explore"""
        result = explorer_model(render_params, hyper_params_history)
        return result

    def on_preview_audio(self, all_state, *list_for_generate):
        print("on_preview_audio")
        if list_for_generate[0]=="":
            mytext = "Hi, this is an example of converting text to audio. This is a bot speaking here, not a real human!"
        else:
            mytext = list_for_generate[0]

        vo,vo_len_sec = create_voice_over_for_storyboard(mytext, 1, DefaultRender.seconds)

        return (all_state,vo)

    async def record_and_render(self, all_state, idx, vote_category: int, *ui_param_state):
        # TODO: use this in on_generate
        #       Going to need to create a new explorer model that uses the ui_param_state to generate random h_params

        cell_params = all_state["im_explorer_hparams"][idx]
        all_state["history"].append((cell_params, vote_category))
        render_params = self.get_sb_multi_sample_params_from_ui(ui_param_state).render
        render_params.batch_size = MAX_IEXP_SIZE
        render_call_args = self.explore(render_params, all_state["history"], SimpleExplorerModel())
        image_results, all_state["im_explorer_hparams"] = await self.render_explorer(render_call_args)

        return all_state, *image_results, *[i.__dict__ for i in all_state["im_explorer_hparams"]]

    async def on_upvote(self, idx, all_state, *ui_param_state):
        return await self.record_and_render(all_state, idx, 1, *ui_param_state)

    async def on_downvote(self, idx, all_state, *ui_param_state):
        return await self.record_and_render(all_state, idx, 2, *ui_param_state)

    async def on_promote(self, idx, all_state, *args):
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
            if all_state["story_board"]==[]:
                all_state["story_board"] = [None]*MAX_SB_SIZE
            try:
                sb_idx: int = all_state["story_board"].index(None)
            except ValueError:
                sb_idx = -1

        if "set simple params":
            all_state["history"].append((cell_params, 4))
            all_state["story_board"][sb_idx]=cell_params

        if "handle the storyboard first":
            # render the new storyboard image
            call_args = self.get_sb_multi_sample_params_from_ui(ui_state_comps)
            call_args.hyper = cell_params
            call_args.render.batch_size = 1
            # re-render the storyboard image
            sb_result = await self.sd_sb_renderer_batched(call_args, 0)
            # update story_board_images
            story_board_images[sb_idx] = sb_result.all_images[0]

            # get new SBMultiSampleArgs for the image explorer
            # render_call_args = self.explore(render_params, all_state["history"], self.SimpleExplorerModel())

        if "handle the regeneration of the explorer second but only if the storyboard is not complete":
            if len([x for x in all_state['story_board'] if x]) < NUM_SB_IMAGES:
                tmp = list(await self.on_generate(all_state, *ui_state_comps))
                all_state, tmp = tmp[0], tmp[1:]
                exp_images, tmp = tmp[:9], tmp[9:]
                exp_texts, _ = tmp[:9], tmp[9:]
            else:
                # todo: find a way to pass something that doesnt change these values
                exp_images, exp_texts = [None]*9, [""]*9

        # TODO: to make this owrk the lists contained in "all_components"
        #  need to be expanded to not be lists and then passed
        #return {self.all_state:all_state,
        #        self.all_components["story_board_images"]:story_board_images,
        #        self.all_components["im_explorer"]["images"]:exp_images,
        #        self.all_components["im_explorer"]["texts"]:exp_texts
        #        }
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
                batch_size=MAX_BATCH_SIZE,
                tiling=ui_param_state[7],
                batch_count=1,
                restore_faces=ui_param_state[6],
                sampler_index=ui_param_state[3],
                sampler_name=DefaultRender.sampler_name,
            )
        )
        return sb_image

    async def on_generate(self, all_state, *ui_param_state):
        """
        generates the first series of images for the user to explore, clears the history, and returns the new params
        :param all_state: the state of the app
        :param ui_param_state: the state of the ui
        :return: the new state of the app, the new storyboard images, the new image explorer images, the new image explorer text
        """

        async def get_random_params_and_images(base_params: SBMultiSampleArgs):
            """
            generates a set of random parameters and renders the images for the image explorer
            """
            print("get_random_params_and_images")
            out_sb_image_hyper_params = []

            def random_pompt_word_weights(prompt_to_randomize: str):
                if run_test:
                    print(f'prompt_to_randomize: {prompt_to_randomize}')
                if ONLY_USE_NOUNS:
                    from  modules.storysquad_storyboard.storyboard import _get_noun_list
                    noun_list = _get_noun_list()

                # if the prompt is a list not a string fix it
                if isinstance(prompt_to_randomize, list) and len(prompt_to_randomize) == 1:
                    prompt_to_randomize = prompt_to_randomize[0]
                elif isinstance(prompt_to_randomize, list) and len(prompt_to_randomize) == 0:
                    prompt_to_randomize = ""

                prompt_to_randomize = prompt_to_randomize.split(" ")

                if ONLY_USE_NOUNS:
                    prompt_to_randomize = ' '.join([w for w in prompt_to_randomize if w in noun_list])
                else:
                    prompt_to_randomize = ' '.join([w for w in prompt_to_randomize])

                words, weights = zip(*get_prompt_words_and_weights_list(prompt_to_randomize))
                weights = [(random.random() - .5) + w for w in weights]
                prompt_to_randomize = " ".join([f"({w}:{weights[i]})" for i, w in enumerate(words)])

                return prompt_to_randomize

            run_test = False
            out_call_args: SBMultiSampleArgs = SBMultiSampleArgs(render=base_params._render, hyper=[])

            if base_params.hyper[0].prompt is None or base_params.hyper[0].prompt == "" or base_params.hyper[0].prompt == []:
                run_test = True
                print("running test")
                base_params._hyper[0].prompt = "this is a test prompt that jumped over a lazy dog and then ran away"
                base_params._hyper[0].negative_prompt = "(mutant :1.2) (confusing :1.2) blurry strange odd two heads (amature:1.2) person (meme:1.2)"

            for i in range(MAX_IEXP_SIZE):
                tmp: SBIHyperParams = copy.deepcopy(base_params._hyper[0])
                tmp_prompt = random_pompt_word_weights(base_params._hyper[0].prompt)

                tmp.prompt = [tmp_prompt]
                tmp.seed = [modules.processing.get_fixed_seed(-1)]

                out_call_args += tmp
                out_sb_image_hyper_params.append(tmp)

            sb_image_results: SBImageResults = self.sd_sb_renderer_s_batch(out_call_args.combined,1)

            out_images = await sb_image_results
            out_images = out_images.all_images

            # remove the image grid from the result if it exists
            if len(out_images) != MAX_IEXP_SIZE:
                out_images = out_images[1:]

            return out_images, out_sb_image_hyper_params

        sb_msp = self.get_sb_multi_sample_params_from_ui(ui_param_state)

        res =  await get_random_params_and_images(sb_msp)
        images_out, params = res
        params:[SBIHyperParams] = params
        text_out = self.update_image_exp_text(params)
        all_state["im_explorer_hparams"] = params
        return all_state, *images_out, *text_out, *[None] * 3

    def get_story_squad_ui(self):
        """
        defines the ui for the story squad app, also sets the initial state of the app
        """
        import gradio as gr
        self.all_components["param_inputs"] = {}



        with gr.Blocks() as param_area:
            with gr.Column(variant='panel'):
                # so what we want here is for when is:
                # dev mode then the value should be 5
                # if product is clash then the slider should not be visible
                # if product is anything else it should be visible

                sl = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=5)
                if sb_env.STORYBOARD_PRODUCT == "clash":
                    sl.visible = False
                elif sb_env.STORYBOARD_PRODUCT != "clash":
                    sl.visible = True
                if sb_env.STORYBOARD_DEV_MODE:
                    sl.value = 5
                self.ordered_list_of_param_inputs.append(sl)

                self.all_components["param_inputs"]["steps"] = self.ordered_list_of_param_inputs[-1]

                self.ordered_list_of_param_inputs.append(gr.State(9))
                self.all_components["param_inputs"]["sampler_index"] = self.ordered_list_of_param_inputs[-1]

                with gr.Row():
                    self.ordered_list_of_param_inputs.append(
                        gr.Slider(minimum=64, maximum=2048, step=64,label="Width",value=512) if DEV_MODE
                        else gr.State(DefaultRender.width))
                    self.all_components["param_inputs"]["width"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(
                        gr.Slider(minimum=64, maximum=2048, step=64,label="Height",value=512) if DEV_MODE
                        else gr.State(DefaultRender.height))
                    self.all_components["param_inputs"]["height"] = self.ordered_list_of_param_inputs[-1]

                with gr.Row():
                    self.ordered_list_of_param_inputs.append(gr.Checkbox(label='Restore faces', value=False,
                                                                         visible=True)if DEV_MODE
                        else gr.State(DefaultRender.restore_faces))
                    self.all_components["param_inputs"]["restore_faces"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(gr.Checkbox(label='Tiling', value=False) if DEV_MODE
                        else gr.State(DefaultRender.tiling))
                    self.all_components["param_inputs"]["tiling"] = self.ordered_list_of_param_inputs[-1]

                with gr.Row():
                    self.ordered_list_of_param_inputs.append(gr.Slider(minimum=1, step=1, label='Batch count', value=1)if DEV_MODE
                        else gr.State(DefaultRender.batch_count))
                    self.all_components["param_inputs"]["batch_count"] = self.ordered_list_of_param_inputs[-1]

                    self.ordered_list_of_param_inputs.append(gr.State(MAX_IEXP_SIZE))
                    self.all_components["param_inputs"]["batch_size"] = self.ordered_list_of_param_inputs[-1]


                if sb_env.STORYBOARD_PRODUCT == "expert":
                    self.ordered_list_of_param_inputs.append(
                        gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0))
                    self.all_components["param_inputs"]["cfg_scale"] = self.ordered_list_of_param_inputs[-1]
                    # seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox
                    self.ordered_list_of_param_inputs.append(self.create_seed_inputs("storyboard"))

                    self.all_components["param_inputs"]["seed"], \
                    self.all_components["param_inputs"]["reuse_seed"], \
                    self.all_components["param_inputs"]["subseed"], \
                    self.all_components["param_inputs"]["reuse_subseed"], \
                    self.all_components["param_inputs"]["subseed_strength"], \
                    self.all_components["param_inputs"]["seed_resize_from_h"], \
                    self.all_components["param_inputs"]["seed_resize_from_w"], \
                    self.all_components["param_inputs"]["seed_checkbox"] = self.ordered_list_of_param_inputs[-8:]

                elif sb_env.STORYBOARD_PRODUCT == "market":
                    self.ordered_list_of_param_inputs.append(
                        gr.Slider(minimum=4.0, maximum=10.0, step=0.1, label='Configuration Scale', value=7.0))
                    self.all_components["param_inputs"]["cfg_scale"] = self.ordered_list_of_param_inputs[-1]
                    self.all_components["param_inputs"]["seed"] = gr.State(DEFAULT_HYPER_PARAMS_DICT.seed)
                    self.all_components["param_inputs"]["subseed"] = gr.State(DEFAULT_HYPER_PARAMS_DICT.subseed)
                    self.all_components["param_inputs"]["subseed_strength"]= gr.State(
                        DEFAULT_HYPER_PARAMS_DICT.subseed_strength)

                elif sb_env.STORYBOARD_PRODUCT == "clash":
                    self.ordered_list_of_param_inputs.append(
                        gr.State(7.0))
                    self.all_components["param_inputs"]["cfg_scale"] = self.ordered_list_of_param_inputs[-1]
                    self.all_components["param_inputs"]["seed"] = gr.State(DEFAULT_HYPER_PARAMS_DICT.seed)
                    self.all_components["param_inputs"]["subseed"] = gr.State(DEFAULT_HYPER_PARAMS_DICT.subseed)
                    self.all_components["param_inputs"]["subseed_strength"] = gr.State(
                        DEFAULT_HYPER_PARAMS_DICT.subseed_strength)

        with gr.Blocks() as story_squad_interface:
            self.comp_helper.story_squad_interface = story_squad_interface

            self.all_state["history"] = []
            self.all_state["im_explorer_hparams"] = []
            self.all_state["story_board"] = []

            self.all_state = gr.State(self.all_state)
            id_part = "storyboard_call_multi"
            label = make_gr_label("StoryBoard by Story Squad")

            with gr.Row(elem_id="toprow") as top_row:
                with gr.Column(scale=80):
                    self.all_components["param_inputs"]["prompt"] = gr.Textbox(label="Prompt",
                                                                               #elem_id=f"{id_part}_prompt",
                                                                               show_label=False,
                                                                               lines=5,
                                                                               placeholder="Prompt"
                                                                               )

                    self.comp_helper.prompt = self.all_components["param_inputs"]["prompt"]
                    self.all_components["param_inputs"]["negative_prompt"] = gr.Textbox(
                        label="Negative prompt",
                        elem_id=f"{id_part}_neg_prompt",
                        show_label=False,
                        lines=1,
                        placeholder="Negative prompt"
                    )
                    self.comp_helper.negative_prompt =\
                        self.all_components["param_inputs"]["negative_prompt"]

                    if sb_env.STORYBOARD_PRODUCT =="clash":
                        self.all_components["param_inputs"]["negative_prompt"].visible = False
                    param_area.render()

                with gr.Column(scale=1):
                    with gr.Box():
                        with gr.Column(scale=1):
                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")

                            self.all_components["submit"] = gr.Button('Generate', elem_id=f"{id_part}_generate",
                                                                          variant='primary')
                            self.comp_helper.generate_button = self.all_components["submit"]

                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")

                            self.all_components["render"] = gr.Button('Render', elem_id=f"{id_part}_render",
                                                                          variant='primary')
                            self.comp_helper.render_button = self.all_components["render"]
                            # hide the render button if we are in a non-dev mode of clash
                            if sb_env.STORYBOARD_PRODUCT == "clash" and not sb_env.STORYBOARD_DEV_MODE:
                                self.all_components["render"].visible = False

                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")

                            self.all_components["dev_render"] = gr.Button('Dev Render', elem_id=f"{id_part}_dev_render")
                            if sb_env.STORYBOARD_PRODUCT=="clash": self.all_components["dev_render"].visible = False
                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")

                            self.all_components["benchmark"] = gr.Button('Benchmark',
                                                                  variant='primary')
                            if sb_env.STORYBOARD_PRODUCT=="clash": self.all_components["benchmark"].visible = False
                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")

                            self.all_components["on_preview_audio"] = gr.Button('Preview Audio')
                            gr.HTML(value="<span style='padding: 20px 20px 20px 20px;'></span>")
                            if sb_env.STORYBOARD_PRODUCT=="clash": self.all_components["on_preview_audio"].visible = False

            with gr.Column():
                gr.HTML("<hr>")
                gr.update()
                with gr.Row(scale=1, variant="panel", elem_id="changeme") as storyboard_display:  # story board
                    with gr.Column():
                        make_gr_label("StoryBoard")
                        with gr.Row(label="StoryBoard", scale=1):
                            image1 = gr.Image(label="position 1", interactive=True,type="pil")
                            image2 = gr.Image(label="position 2", interactive=True,type="pil")
                            image3 = gr.Image(label="position 3", interactive=True,type="pil")
                            for i,img in enumerate([image1, image2, image3]):
                                img.change(self.drag_image_in,
                                           inputs=[
                                               gr.State(i),
                                               img,
                                               self.all_state,
                                               self.comp_helper.prompt,
                                               self.comp_helper.negative_prompt
                                           ],
                                           outputs=[self.all_state,
                                                    img,
                                                    self.comp_helper.prompt,
                                                    self.comp_helper.negative_prompt
                                                    ]
                                           )

                            self.all_components["story_board_images"] = [image1, image2, image3]



                def refresh_renders_display(idx):
                    video_file = get_files_at_path(path=sb_env.STORYBOARD_RENDER_PATH)
                    if idx>=len(video_file)-1:
                        return None
                    if isinstance(video_file,list):
                        video_file=video_file[idx]
                    else:
                        video_file = video_file
                    out = videos_comps[idx].update(visible=True, value=video_file)
                    return out

                with gr.Blocks() as interfaces:
                    with gr.Column() as robot_dreams:
                        self.comp_helper.robot_dreams = robot_dreams
                        gr.HTML("<hr>")
                        make_gr_label("RobotDreams")
                        with gr.Row(scale=1) as image_explorer:
                            self.comp_helper.image_explorer = image_explorer
                            with gr.Blocks():
                                for r in range(3):
                                    with gr.Row(equal_height=True):
                                        for c in range(3):
                                            with gr.Column(equal_width=True):
                                                with gr.Group():
                                                    self.create_img_exp_group()

                        with gr.Row(scale=1) as movie_result:
                            self.all_components["story_board_render"] = gr.Video()
                            self.all_components["story_board_audio"]= gr.Audio()

                            if sb_env.STORYBOARD_PRODUCT=="clash":
                                self.all_components["story_board_render"].visible = False
                                self.all_components["story_board_audio"].visible = False

                    with gr.Column() as files_interface:
                        self.comp_helper.files_interface = files_interface

                        files_interface.visible = False
                        # files_tab.select(lambda: gr.Tabs.update(selected=1), None, interface_tabs)
                        videos_comps = []
                        with gr.Tabs() as file_tabs:
                            for tbnum in range(10):
                                with gr.TabItem(str(tbnum), id=tbnum) as tab:
                                    # tab.select(lambda x: set_tab(x), gr.State(tbnum), file_tabs)
                                    with gr.Row() as pg:
                                        for _ in range(10):
                                            with gr.Column(scale=0.10, min_width=64):
                                                with gr.Row():
                                                    videos_comps.append(gr.Video(
                                                        interactive=False, label=None,
                                                        show_label=False, elem_id="video_file", visible=False
                                                        ))
                    with gr.Column() as loading_please_wait:
                        loading_please_wait.visible = False
                        with gr.Row():
                            gr.HTML("<h1>Rendering, please wait...</h1>")

            self.all_components["param_inputs"]["list_for_generate"] = \
                [self.all_components["param_inputs"][k] for k in
                 keys_for_ui_in_order]

            render_main_func_kwargs = {"fn":self.on_render_storyboard,
                                "inputs":[gr.State(DefaultRender.early_stop_seconds),
                                        self.all_state,
                                        *self.all_components["param_inputs"]["list_for_generate"]
                                        ],
                                "outputs":[self.all_state, self.all_components["story_board_render"]]}

            generate_main_func_kwargs = {"fn":self.on_generate,
                                    "inputs" : [self.all_state,
                                                *self.all_components["param_inputs"]["list_for_generate"]
                                                ],
                                    "outputs" : [self.all_state,
                                               *self.all_components["im_explorer"]["images"],
                                               *self.all_components["im_explorer"]["texts"],
                                               *self.all_components["story_board_images"]
                                               ]
                                    }


            if sb_env.STORYBOARD_PRODUCT == "expert":
                # GENERATE BUTTON
                self.all_components["submit"].click(**generate_main_func_kwargs)
                # SIMULATE BUTTON
                if "create a button to simulate what happens when the render is done":
                    sim_rc = gr.Button("Simulate Render Complete")
                    sim_rc.click(fn=lambda: gr.Tabs.update(selected=0), outputs=[file_tabs])
                    sim_rc.click(fn=lambda: gr.update(visible=False), outputs=[robot_dreams])
                    sim_rc.click(fn=lambda: gr.update(visible=True), outputs=[files_interface])

                    for i in range(100):
                        sim_rc.click(fn=refresh_renders_display,
                                inputs=gr.State(i),
                                outputs=videos_comps[i:(i + 1)])

                    sim_rc.click(fn=lambda: gr.update(visible=False), outputs=[sim_rc])

                # RENDER BUTTON
                self.all_components["render"].click(**render_main_func_kwargs)

            elif sb_env.STORYBOARD_PRODUCT == "clash":
                if "this sections enoforces the user-flow for non-dev mode":
                    if "events for the render button":
                        queue_option = False
                        # part one is to hide everything except the loading gif, this can be done all at once
                        # part two is to generate the movie and then shwo the file explorer, this has to be done sequentially
                        # we assume that if part one and part two are executed at the same time
                        # then the user will see the loading gif and then the file explorer once the movie is generated

                        def render_pressed_part_one():
                            return {
                                top_row:gr.update(visible=False),
                                storyboard_display: gr.update(visible=False),
                                robot_dreams: gr.update(visible=False),
                                loading_please_wait: gr.update(visible=True),
                            }

                        def render_pressed_part_two(data):
                            ongen_data= [data[i] for i in self.all_components["param_inputs"]["list_for_generate"]]
                            all_state,complete_mp4_f_path = \
                                self.on_render_storyboard(DefaultRender.early_stop_seconds,data[self.all_state] , *ongen_data)

                            vc_update = {c:refresh_renders_display(i) for i,c in enumerate(videos_comps)}

                            out = {
                                self.all_state: all_state,
                                files_interface: gr.update(visible=True),
                                loading_please_wait: gr.update(visible=False),
                            }
                            out.update(vc_update)
                            return out

                        self.all_components["render"].click(fn=render_pressed_part_one,
                                                            outputs={top_row,storyboard_display,robot_dreams,loading_please_wait}
                                                            )
                        self.all_components["render"].click(fn=render_pressed_part_two,
                                                            inputs={self.all_state,
                                                                    *self.all_components["param_inputs"]["list_for_generate"],
                                                                    *videos_comps
                                                                    },
                                                            outputs={self.all_state,files_interface,
                                                                     *videos_comps,
                                                                     loading_please_wait
                                                                     }
                                                            )
                    if "events for the generate button":
                        gen_button =self.all_components["submit"]
                        gen_button.click(**generate_main_func_kwargs)
                        gen_button.click(fn=lambda: gr.update(visible=False),
                        outputs=[self.all_components["submit"]])
                    if "events for use/promote to storyboard button":
                        for b in self.all_components["im_explorer"]["buttons"]:

                            #event that manages robot dreams and the render and generate buttons
                            if "use" in b.value.lower():
                                b.click(
                                    lambda x: [gr.update(visible=True), #dreams
                                               gr.update(visible=False),#generate
                                               gr.update(visible=False) #render
                                               ] if len(x["story_board"]) != 2
                                    else [gr.update(visible=False), #dreams
                                          gr.update(visible=False), #generate
                                          gr.update(visible=False)  #render
                                          ],
                                    inputs=self.all_state,
                                    outputs=[self.comp_helper.robot_dreams,
                                             self.comp_helper.generate_button,
                                             self.comp_helper.render_button
                                             ]
                                )





        return story_squad_interface

    def setup_story_board_events(self):
        import gradio as gr
        """
        Setup the events for the story board using self.all_components that was initialized in __init__
        which called this function.
        """


        self.all_components["param_inputs"]\
            ["prompt"].change(self.reset_story_board,
                         inputs ={
                             self.all_state,
                             *self.all_components["story_board_images"],
                             *self.all_components["im_explorer"]["images"],
                             self.comp_helper.prompt
                         },
                         outputs={
                             self.all_state,
                             *self.all_components["story_board_images"],
                             *self.all_components["im_explorer"]["images"],
                             *self.all_components["im_explorer"]["texts"],
                             self.comp_helper.robot_dreams,
                             self.comp_helper.render_button,
                             self.comp_helper.generate_button,
                            }
                         )


        self.all_components["dev_render"].click(self.on_render_storyboard_dev,
                                            inputs=[gr.State(DefaultRender.early_stop_seconds),
                                                    self.all_state,
                                                    *self.all_components["param_inputs"]["list_for_generate"]
                                                    ],
                                            outputs=[self.all_state, self.all_components["story_board_render"]]
                                            )


        self.all_components["benchmark"].click(self.render_storyboard_benchmark,
                                            inputs=[self.all_state,
                                                    *self.all_components["param_inputs"]["list_for_generate"]
                                                    ],
                                            outputs=[self.all_state, self.all_components["story_board_render"]]
                                            )
        self.all_components["on_preview_audio"].click(self.on_preview_audio,
                                                   inputs=[self.all_state,
                                                    *self.all_components["param_inputs"]["list_for_generate"]
                                                    ],
                                                   outputs=[self.all_state, self.all_components["story_board_audio"]]
                                                   )



        # forced workflow events
        if "unhide robot dreams if generate has been pressed"=="":
            b = self.all_components["submit"]
            b.click(fn=lambda: gr.update(visible=True), outputs=[robot_dreams])

        # create the events for the image explorer, buttons, and text boxes
        cur_img_idx = 0
        while True:
            # cur_img = self.all_state["im_explorer"]["images"][cur_img_idx]
            but_idx_base = cur_img_idx * 3
            but_up = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                      "up" in b.value.lower()][0]
            but_use = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                       "use" in b.value.lower()][0]
            but_down = [b for b in self.all_components["im_explorer"]["buttons"][but_idx_base:but_idx_base + 3] if
                        "down" in b.value.lower()][0]

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
                                   *self.all_components["im_explorer"]["texts"],
                                   ]
                          )
            but_use.click(
                lambda x: gr.update(visible=True) if len(x["story_board"]) == 2 else gr.update(visible=False),
                inputs=self.all_state,
                outputs=[self.all_components["render"]]
            )
            but_use.click(
                lambda x: gr.update(visible=True) if len(x["story_board"]) == 2 else gr.update(visible=False),
                inputs=self.all_state,
                outputs=[self.all_components["render"]]
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
        import gradio as gr
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
            #with gr.Group(equal_width=True):
            but = gr.Button(f"Up vote")
            gr_comps["im_explorer"]["buttons"].append(but)
            but = gr.Button(f"Use in storyboard")
            gr_comps["im_explorer"]["buttons"].append(but)

            but = gr.Button(f"Down vote")
            gr_comps["im_explorer"]["buttons"].append(but)
            if sb_env.STORYBOARD_PRODUCT=="clash":
                but.visible = False

        text = gr.Textbox(show_label=False, max_lines=3, interactive=False)
        if sb_env.STORYBOARD_PRODUCT=="clash":
            text.visible = False

        gr_comps["im_explorer"]["texts"].append(text)


class StorySquadExtraGradio(StoryBoardGradio):

    @staticmethod
    def dict_to_all_state(new_state: dict,all_state={}):
        # update the all_state dict with the new values
        for key, value in new_state.items():
            all_state[key] = []
            if key =="history":
                for t in value:
                    sbh= SBIHyperParams(**t[0])
                    vote = t[1]
                    all_state[key].append((sbh,vote))
            elif key == "im_explorer_hparams":
                for data in value:
                    all_state[key].append(SBIHyperParams(**data))
            elif key == "story_board":
                for data in value:
                    all_state[key].append(SBIHyperParams(**data))
            else:
                all_state[key] = value
        return all_state
    def all_state_to_dict(self,all_state):

        o = "{"
        for sk, sv in all_state.items():
            o = o + f'"{sk}":'
            o = o + f'['
            for li in sv:
                if type(li) == tuple:
                    o = o + f'{(li[0].__dict__, li[1])},'
                else:
                    o = o + f'{li.__dict__},'
            o = o + "],"
        o = o + "}"
        return o
    def all_state_to_json(self,all_state):
        return json.dumps(self.all_state_to_dict(all_state))
    @staticmethod
    def load_last_state():
        # load the last state
        with open("last_state.json", "r") as f:
            last_state = json.load(f)
        return StorySquadExtraGradio.dict_to_all_state(last_state)

import unittest
class TestStoryboardGradio(unittest.TestCase):
    @staticmethod
    def test_drag_image_in():
        from modules.storysquad_storyboard.storyboard import SBIHyperParams

        sb = StoryBoardGradio()
        idx = 3
        img = Image.new('RGB', (50, 50), color='white')  # Creating a blank white image
        all_state = {"story_board": [SBIHyperParams(), SBIHyperParams(), SBIHyperParams()]}
        prompt = "Once upon a time..."
        negative_prompt = "Do not use any magic."
        updated_state, updated_img, updated_prompt, updated_negative_prompt = sb.drag_image_in(idx, img, all_state, prompt, negative_prompt)

if __name__ == '__main__':
    import doctest
    import random
    import gradio as gr
    import numpy as np
    from typing import List
    from collections import OrderedDict
    from modules.storysquad_storyboard.storyboard import SBIHyperParams

    from PIL import Image
    from unittest.mock import MagicMock
    class MockStoryBoardGradio(StoryBoardGradio):
        def __init__(self):
            self.drag_image_in = StoryBoardGradio.drag_image_in


    StoryBoardGradio.__init__ = MagicMock(return_value=None)

    unittest.main()
    doctest.run_docstring_examples(StorySquadExtraGradio.on_render_storyboard, globals(), verbose=True)



