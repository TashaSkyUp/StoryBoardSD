import copy
import json
import numpy as np
import os
import numbers
from dataclasses import dataclass
from typing import List, Any

DEV_MODE = os.getenv("STORYBOARD_DEV_MODE", "False") == "True"
DEFAULT_HYPER_PARAMS_DICT = {
    "prompt": "",
    "negative_prompt": "",
    "steps": 8,
    "seed": -1,
    "subseed": 0,
    "subseed_strength": 0,
    "cfg_scale": 7,
}


@dataclass
class DEFAULT_HYPER_PARAMS:
    prompt: str = DEFAULT_HYPER_PARAMS_DICT["prompt"]
    negative_prompt: str = DEFAULT_HYPER_PARAMS_DICT["negative_prompt"]
    steps: int = DEFAULT_HYPER_PARAMS_DICT["steps"]
    seed: int = DEFAULT_HYPER_PARAMS_DICT["seed"]
    subseed: int = DEFAULT_HYPER_PARAMS_DICT["subseed"]
    subseed_strength: int = DEFAULT_HYPER_PARAMS_DICT["subseed_strength"]
    cfg_scale: int = DEFAULT_HYPER_PARAMS_DICT["cfg_scale"]


DEV_HYPER_PARAMS = DEFAULT_HYPER_PARAMS(steps=1)

if DEV_MODE:
    DEFAULT_HYPER_PARAMS_DICT = DEV_HYPER_PARAMS
else:
    DEFAULT_HYPER_PARAMS_DICT = DEV_HYPER_PARAMS


class UniqueList:
    def __init__(self):
        self._list = []
        self._set = set()

    def append(self, item):
        if item not in self._set:
            self._list.append(item)
            self._set.add(item)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index, value):
        if value not in self._set:
            self._set.discard(self._list[index])
            self._list[index] = value
            self._set.add(value)

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return repr(self._list)


class SBIHyperParams:
    """
    the idea with this class is to provide a useful interface for the user to set,add,index the hyper parameters
    """

    # def __init__(self, negative_prompt=DEFAULT_HYPER_PARAMS_DICT.negative_prompt,
    #             steps=DEFAULT_HYPER_PARAMS_DICT.steps,
    #             seed=DEFAULT_HYPER_PARAMS_DICT.seed,
    #             subseed=DEFAULT_HYPER_PARAMS_DICT.subseed,
    #             subseed_strength=DEFAULT_HYPER_PARAMS_DICT.subseed_strength,
    #             cfg_scale=DEFAULT_HYPER_PARAMS_DICT.cfg_scale,
    #             prompt=DEFAULT_HYPER_PARAMS_DICT.prompt,
    #             **kwargs):

    def __init__(self,
                 negative_prompt=[],
                 steps=[],
                 seed=[],
                 subseed=[],
                 subseed_strength=[],
                 cfg_scale=[],
                 prompt=[],
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
        ret = SBIHyperParams(prompt=self._prompt[item], negative_prompt=self.negative_prompt[item],
                             steps=self.steps[item], seed=self.seed[item], subseed=self.subseed[item],
                             subseed_strength=self.subseed_strength[item], cfg_scale=self.cfg_scale[item])
        return ret

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
                negative_prompt=self._make_list(self.negative_prompt) + self._make_list(other.negative_prompt),
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

    def __len__(self):
        return len(self._prompt)


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
    >>> get_prompt_words_and_weights_list(f'test list:\\n\\thello:1 world:.2 how are (you:1.0)')
    [('test', 1.0), ('list', 1.0), ('hello', 1.0), ('world', 0.2), ('how', 1.0), ('are', 1.0), ('you', 1.0)]
    >>> get_prompt_words_and_weights_list("")
    Traceback (most recent call last):
        ...
    ValueError: prompt cannot be empty
    >>> get_prompt_words_and_weights_list("hello")
    [('hello', 1.0)]
    >>> get_prompt_words_and_weights_list("hello:0.5")
    [('hello', 0.5)]
    >>> get_prompt_words_and_weights_list("hello:.5 world")
    [('hello', 0.5), ('world', 1.0)]
    >>> get_prompt_words_and_weights_list("hello:5 world:0.2")
    [('hello', 5.0), ('world', 0.2)]
    >>> get_prompt_words_and_weights_list("hello:5 (world:0.2) how (are) you")
    [('hello', 5.0), ('world', 0.2), ('how', 1.0), ('are', 1.0), ('you', 1.0)]
    """
    if prompt == "": raise ValueError("prompt cannot be empty")
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
            if len(word_weight_pair[1]) == 0:  # weight is empty
                w = 1.0
            else:
                # if the second item in the word weight pair is a float then it is a weight
                try:
                    w = float(word_weight_pair[1])
                # if the second item in the word weight pair is not a float then it is not a weight
                except ValueError:
                    print("Could not convert {word_weight_pair[1]} to a float")
                    w = 1.0
        else:
            print(f"Could not convert {word_weight_pair[1]} to a float")
            w = 1.0
        out.append((word_weight_pair[0], w))
    return out


def get_prompt_words_list(prompt):
    out = [i[0] for i in get_prompt_words_and_weights_list(prompt)]
    return out


def get_prompt_words_and_weights_list_new(prompt) -> List[List[str]]:
    """
    >>> get_prompt_words_and_weights_list_new("")
    []
    >>> get_prompt_words_and_weights_list_new("hello")
    [('hello', 1.0)]
    >>> get_prompt_words_and_weights_list_new("hello:0.5")
    [('hello', 0.5)]
    >>> get_prompt_words_and_weights_list_new("hello:.5 world")
    [('hello', 0.5), ('world', 1.0)]
    >>> get_prompt_words_and_weights_list_new("hello:5 world:0.2")
    [('hello', 5.0), ('world', 0.2)]
    >>> get_prompt_words_and_weights_list_new("hello:5 (world:0.2) how (are) you")
    [('hello', 5.0), ('world', 0.2), ('how', 1.0), ('are', 1.0), ('you', 1.0)]
    """
    prompt = sanitize_prompt(prompt)
    words = prompt.split(" ")
    possible_word_weight_pairs = [i.split(":") for i in words]
    return [
        (word_weight_pair[0], float(word_weight_pair[1]))
        if len(word_weight_pair) == 2
        else (word_weight_pair[0], 1.0)
        for word_weight_pair in possible_word_weight_pairs
        if word_weight_pair[0] != ""
    ]


def sanitize_prompt(prompt):
    prompt = prompt.replace(",", " ").replace(". ", " ").replace("?", " ").replace("!", " ").replace(";", " ")
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("\r", " ")
    prompt = prompt.replace("\t", " ")
    prompt = prompt.replace("[", " ").replace("]", " ")
    prompt = prompt.replace("{", " ").replace("}", " ")
    # compact blankspace
    for i in range(10):
        prompt = prompt.replace("  ", " ")

    prompt = prompt.replace("(", "").replace(")", "")
    return prompt.strip()


from typing import List, Tuple


def _get_noun_list() -> List[str]:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("nounlist.csv", 'r') as f:
        noun_list = f.read().splitlines()
    return noun_list


class Interpolatable:
    def __init__(self, start, end, progress, interfunc=None):
        self.x = start
        self.y = end
        self.p = progress

        if interfunc is None:
            self.interfunc = lambda x, y, p: x + (y - x) * p

        if progress > 1 or progress < 0:
            raise ValueError(f"progress must be between 0 and 1; got: {progress}")
        self.progress = progress

    def __call__(self, p):
        return self.interfunc(self.x, self.y, p)

    def __add__(self, other):
        if isinstance(other, float) and 0 <= other <= 1:
            # add progress
            return StoryBoardSeed.InteropSeed(self.x, self.y, self.progress + other)
        else:
            raise ValueError(f"Cannot add {type(other)} to {type(self)}, only float between 0 and 1")

    def __sub__(self, other):
        if isinstance(other, float) and 0 <= other <= 1:
            # subtract progress
            return StoryBoardSeed.InteropSeed(self.x, self.y, self.progress - other)
        else:
            raise ValueError(f"Cannot subtract {type(other)} from {type(self)}, only float between 0 and 1")

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.progress})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.progress == other.progress

    def __hash__(self):
        return hash((self.x, self.y, self.progress))


class StoryBoardPrompt:
    """
    A prompt for storyboard consists of all the words for every section of the storyboard, and a function to sample
    each words weight as some percentage of completion for that section. for instance if we want a story board of a dog
    transforming into a cat, we might have a prompt like this:

    ["dog :1 cat:0.0",
    "dog :1. cat:1.0",
    "dog :0. cat: 1"]

    notice that each section has the same words, but the weights for each word are different.
    the first section has a weight of 1 for dog, and 0 for cat
    the second section has a weight of 1 for dog, and 1 for cat
    the third section has a weight of 0 for dog, and 1 for cat

    this provides a way to render the prompt attribute of a story board at a specified point in time

    seconds_lengths is a list of the lengths of each section of the storyboard in seconds
    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB_part1 = SB[.25:.75] #.5 seconds sliced out of the middle
    ...     detailed= f'{SB_part1([0.0])}{SB_part1([0.125])}{SB_part1([0.25+0.125])}{SB_part1([0.5])}'
    ...     print(detailed)
    ... except Exception as e:
    ...     raise e
    ['(dog:1.00000000)(cat:0.50000000)']['(dog:1.00000000)(cat:0.75000000)']['(dog:0.75000000)(cat:1.00000000)']['(dog:0.50000000)(cat:1.00000000)']
    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB_part1 = SB[0:.5]
    ...     SB_part1([0.25])
    ... except Exception as e:
    ...     raise e
    ['(dog:1.00000000)(cat:0.50000000)']
    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB_part2 = SB[.25:]
    ...     SB_part2([0.5])
    ... except Exception as e:
    ...     raise e
    ['(dog:0.50000000)(cat:1.00000000)']
    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB._get_prompt_at_time(0.5)
    ... except Exception as e:
    ...     print(e)
    '(dog:1.00000000)(cat:1.00000000)'
    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB([0.5])
    ... except Exception as e:
    ...     print(e)
    ['(dog:1.00000000)(cat:1.00000000)']
    """

    def __init__(self, prompts: List[str] or str, seconds_lengths: List[float], use_only_nouns=False):
        self.noun_list = _get_noun_list()

        self._prompts = prompts
        if isinstance(seconds_lengths, list):
            self.seconds_lengths = seconds_lengths
            self.total_seconds = sum(seconds_lengths)
        elif isinstance(seconds_lengths, numbers.Number):
            secs_per_prompt = seconds_lengths / (len(prompts) - 1)
            self.seconds_lengths = [secs_per_prompt] * (len(prompts) - 1)
            self.total_seconds = seconds_lengths

        self._testing_dirty_prompts = [
            "dog :1 cat:0.0",
            "dog :1. cat:1.0",
            "dog :0. cat: 1",
        ]
        self._times_sections_start = [sum(self.seconds_lengths[:i]) for i in range(len(self.seconds_lengths))]

        if prompts == "doctests":
            self._prompts = self._testing_dirty_prompts

        self._sanitized_prompts = [self._sanitize_prompt(p) for p in self._prompts]

        self._words_and_weights = [self._get_prompt_words_and_weights_list(p) for p in
                                   self._sanitized_prompts]
        if use_only_nouns:
            self._words_and_weights = [self._get_nouns_only(p) for p in self._words_and_weights]

        self._sections = self._get_sections(self._words_and_weights)

        self._frame_values = self._get_frame_values_for_prompt_word_weights(self._sections, 4)

        self._words = [w[0] for w in self._words_and_weights[0]]

    def __call__(self, *args, **kwargs) -> [str]:
        # if args is a single float, then we are getting the prompt at a specific point in time
        # if args is a list of floats, then we are getting the prompt at a list of points in time
        try:
            if isinstance(args[0], numbers.Number):
                return [self._get_prompt_at_time(args[0])]
            elif isinstance(args[0], np.floating):
                return [self._get_prompt_at_time(args[0])]
            elif type(args[0]) is list:
                return [self._get_prompt_at_time(t) for t in args[0]]
            elif type(args[0]) is slice:
                return self._slice(args[0])
            else:
                raise TypeError("args must be a float or a list of floats")
        except IndexError as e:
            raise e

    def _slice(self, slice: slice):
        start_time = slice.start
        stop_time = slice.stop
        if start_time is None:
            start_time = 0
        if stop_time is None:
            stop_time = 1
        start_prompt = self(start_time)[0]  # this returns a prompt, one of the new helix points
        stop_prompt = self(stop_time)[0]  # this returns a prompt, one of the new helix points

        # check if a helix point is contained between start and stop
        # if so, then we need to include it in the output StoryBoardPrompt
        # if not, then we can just return a StoryBoardPrompt with two prompts and one section
        helix_point_times: [float] = self._times_sections_start

        helix_point_prompts: [str] = [
            start_prompt,
        ]
        new_section_lengths: [float] = []
        for i in range(len(helix_point_times)):
            if start_time < helix_point_times[i] < stop_time:
                helix_point_prompts.append(self._get_prompt_at_time(helix_point_times[i]))
        helix_point_prompts.append(stop_prompt)

        new_start_section_length = helix_point_times[1] - start_time
        new_end_section_length = stop_time - helix_point_times[-1]

        if new_end_section_length == 0 or new_start_section_length == 0:
            num_sections = 1
        else:
            num_sections = len(helix_point_prompts)

        if num_sections == 1:
            new_section_lengths = [stop_time - start_time]
        else:
            new_section_lengths = [new_start_section_length] + new_section_lengths + [new_end_section_length]
        ret = StoryBoardPrompt(helix_point_prompts, new_section_lengths)
        return ret

    @staticmethod
    def _sanitize_prompt(prompt):
        """
        >>> StoryBoardPrompt._sanitize_prompt("dog :1 cat:.0")
        'dog:1 cat:.0'
        """
        prompt = prompt.replace(" :", ":").replace(" .", ".").replace(" ,", ",").replace(" ?", "?")
        prompt = prompt.replace(": ", ":")
        prompt = prompt.replace("( ", "(")
        prompt = prompt.replace(" )", ")")
        prompt = prompt.replace(",", " ").replace(". ", " ").replace("?", " ").replace("!", " ").replace(";", " ")
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("\r", " ")
        prompt = prompt.replace("[", " ").replace("]", " ")
        prompt = prompt.replace("{", " ").replace("}", " ")
        # compact blankspace
        for i in range(10):
            prompt = prompt.replace("  ", " ")
        # removed for now because it prevents slicing the object
        # prompt = prompt.replace("(", "").replace(")", "")
        prompt = prompt.replace(")(", " ")
        prompt = prompt.replace("(", "").replace(")", "")
        return prompt.strip()

    @staticmethod
    def _get_prompt_words_and_weights_list(prompt) -> List[Tuple[str, float]]:
        words = prompt.split(" ")
        possible_word_weight_pairs = [i.split(":") for i in words]

        out: List[Tuple[str, float]] = []
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

    @staticmethod
    def _get_sections(words_and_weights_list: List[List[Tuple[str, float]]]) -> List[List[List[Tuple[str, float]]]]:
        """
        >>> try:
        ...     SB = StoryBoardPrompt("doctests", [0.5, 0.5])
        ...     SB._get_sections(SB._words_and_weights)
        ... except Exception as e:
        ...     print(e)
        [[[('dog', 1.0), ('cat', 0.0)], [('dog', 1.0), ('cat', 1.0)]], [[('dog', 1.0), ('cat', 1.0)], [('dog', 0.0), ('cat', 1.0)]]]
        >>> try:
        ...     SB = StoryBoardPrompt(["doctests:0","doctest:1"], [0.5])
        ...     SB._get_sections(SB._words_and_weights)
        ... except Exception as e:
        ...     print(e)
        [[[('doctests', 0.0)], [('doctest', 1.0)]]]
        """
        sections: List[List[List[Tuple[str, float]]]] = []
        for i in range(len(words_and_weights_list) - 1):
            sections.append([words_and_weights_list[i], words_and_weights_list[i + 1]])

        return sections

    @staticmethod
    def _get_word_weight_at_percent(section, word_index, percent):
        """
        >>> try:
        ...     SB = StoryBoardPrompt("doctests", [0.5, 0.5])
        ...     SB._get_word_weight_at_percent(SB._sections[0], 0, 0.5)
        ... except Exception as e:
        ...     print(e)
        1.0
        """
        start_weight = section[0][word_index][1]
        end_weight = section[1][word_index][1]
        return start_weight + percent * (end_weight - start_weight)

    @staticmethod
    def _get_frame_values_for_prompt_word_weights(sections,
                                                  num_frames):  # list[sections[frames[word:weight tuples]]]
        """
        >>> while True:
        ...     SB = StoryBoardPrompt("doctests", [.5,.5])
        ...     sections = SB._get_frame_values_for_prompt_word_weights(SB._sections, num_frames=4)
        ...     for section in sections:
        ...         print(section)
        ...     break
        [[('dog', 1.0), ('cat', 0.0)], [('dog', 1.0), ('cat', 0.3333333333333333)], [('dog', 1.0), ('cat', 0.6666666666666666)], [('dog', 1.0), ('cat', 1.0)]]
        [[('dog', 1.0), ('cat', 1.0)], [('dog', 0.6666666666666667), ('cat', 1.0)], [('dog', 0.33333333333333337), ('cat', 1.0)], [('dog', 0.0), ('cat', 1.0)]]
        """
        # get the weights for each word of each prompt in the prompts list returns a list of lists of tuples
        # words_and_weights_for_prompts = [StoryBoardPrompt._get_prompt_words_and_weights_list(p) for p in prompts]

        # define the two distinct sections of the storyboard_call_multi
        # each section is composed of frames, each frame has different weights for each word (probably) which results
        # in a unique image for the animation

        # contents= sections[words[word,weight]]

        sections_frames = []
        for section in sections:
            start: List[tuple[str, float]] = section[0]
            end: List[tuple[str, float]] = section[1]
            word_frame_weights = []
            for i in range(num_frames):
                frame_weights = []
                for word_idx, word_at_pos in enumerate(start):
                    # format like: ('dog', 0.0)
                    word_frame_weight = StoryBoardPrompt._get_word_weight_at_percent(section, word_idx,
                                                                                     i / (num_frames - 1))

                    frame_weights.append((word_at_pos[0], word_frame_weight))
                word_frame_weights.append(frame_weights)
            sections_frames.append(word_frame_weights)

        return sections_frames

    def _get_prompt_at_time(self, seconds: float, *args):
        """
        >>> try:
        ...     SB = StoryBoardPrompt("doctests",[0.5,0.5])
        ...     SB._get_prompt_at_time(0.0)
        ... except Exception as e:
        ...     raise e
        '(dog:1.00000000)(cat:0.00000000)'
        >>> try:
        ...     SB = StoryBoardPrompt("doctests",[0.5,0.5])
        ...     SB._get_prompt_at_time(0.5)
        ... except Exception as e:
        ...     raise e
        '(dog:1.00000000)(cat:1.00000000)'
        >>> try:
        ...     SB = StoryBoardPrompt("doctests",[0.5,0.5])
        ...     SB._get_prompt_at_time(0.75)
        ... except Exception as e:
        ...     raise e
        '(dog:0.50000000)(cat:1.00000000)'

        """

        if seconds < 0:
            raise ValueError("seconds cannot be less than 0")
        if seconds > self.total_seconds:
            raise ValueError("seconds cannot be greater than the total time of the storyboard")

        # find the section that the seconds is in using self._times_sections_start
        section_second_is_in = self._get_section_at_time(seconds)

        section_start_time = self._times_sections_start[section_second_is_in]
        section_end_time = self._times_sections_start[section_second_is_in] + self.seconds_lengths[section_second_is_in]
        section_length = section_end_time - section_start_time
        section_percent = (seconds - section_start_time) / section_length

        prmpt = []
        section_data = self._sections[section_second_is_in]
        for w_idx, word in enumerate(self._words):
            prmpt.append([word,
                          self._get_word_weight_at_percent(
                              section_data,
                              w_idx,
                              section_percent)
                          ])

        return "".join([f"({w[0]}:{w[1]:.8f})" for w in prmpt])

    def _get_section_at_time(self, seconds):
        section_second_is_in = len(self._times_sections_start) - 1
        for i, section_start_time in enumerate(self._times_sections_start):
            if seconds < section_start_time:
                section_second_is_in = i - 1
                break
        return section_second_is_in

    def __getitem__(self, time_seconds: float) -> str:
        return self(time_seconds)

    def _get_nouns_only(self, p):
        out = [sp for sp in p if sp[0] in self.noun_list]
        return out


class StoryBoardSeed:
    """
    The StoryBoardSeed class represents a storyboard containing a sequence of seeds and their corresponding times. It manages the transition between seeds and provides methods to retrieve prime seeds, sub seeds, and their interpolation progress at specified times. The class can be used to manage and manipulate a collection of Seed objects and their transitions over time.

    Attributes:
    seeds (List[int]): A list of seed values.
    times (List[int]): A list of times corresponding to each seed value.
    seeds and times are conceptualized as sections:
    section[i] = (start_seed = seeds[i],
                  start_time = times[i],
                  end_seed = seeds[i+1],
                  end_time = times[i+1])
    this gives us len(seeds) - 1 total number of sections

    Methods:
    get_seed_at_time(seconds): Returns the seed value at the specified time.
    get_sub_seed_at_time(seconds): Returns the sub seed value at the specified time.
    get_progress_at_time(seconds): Returns the interpolation progress at the specified time.

    """

    class InteropSeed(Interpolatable):
        def __init__(self, seed_x, seed_y, progress, interfunc=None):
            if progress > 1 or progress < 0:
                raise ValueError(f"progress must be between 0 and 1; got: {progress}")

            super().__init__(seed_x, seed_y, progress, interfunc=interfunc)
            self.interfunc = lambda x, y, p: self.__class__(x, y, p)

        def __repr__(self):
            return f"InteropSeed({self.x}, {self.y}, {self.progress})"

        def __str__(self):
            return f"InteropSeed({self.x}, {self.y}, {self.progress})"

    class TimeSection(Interpolatable):
        def __init__(self, start_time, end_time, interfunc=None):
            super().__init__(start_time, end_time, 0, interfunc=interfunc)

        def __contains__(self, item):
            if issubclass(item.__class__, Interpolatable):
                return self.x <= item.x <= self.y and self.x <= item.y <= self.y

            elif isinstance(item, (float,int)):
                return self.x <= item <= self.y

            else:
                raise ValueError(f"TimeSection.__contains__ only supports Interpolatable and float; got: {type(item)}")

        def __repr__(self):
            return f"TimeSection({self.x}, {self.y})"

        def __str__(self):
            return f"TimeSection({self.x}, {self.y})"

    class InterpolatableSection:
        def __init__(self, i_seed_start: Interpolatable, i_seed_end: Interpolatable, intr_time: Interpolatable):
            self.i_seed_start = i_seed_start
            self.i_seed_end = i_seed_end
            if not self.i_seed_start.y == self.i_seed_end.x:
                if i_seed_start.y == i_seed_end.y and i_seed_start.x == i_seed_end.x:
                    pass
                else:
                    raise ValueError(
                        f"i_seed_start.y must equal i_seed_end.x; got: {self.i_seed_start.y} != {self.i_seed_end.x}"
                    )
            self.i_time = intr_time

        def __call__(self, time):
            if not isinstance(time, (float, int)):
                raise ValueError(f"Time must be a float or int; got: {type(time)}")

            if time in self.i_time:
                return self.get_i_seed(time)

            else:
                raise ValueError(f"Time is not in the interpolatable section: {time}")

        def get_i_seed(self, time):

            # find the percent of the time that has passed
            time_into_section_per = (time - self.i_time.x) / (self.i_time.y - self.i_time.x)

            # seeds are arranged as (a,b,p1) (b,c,p2)
            # meaning that there is some point between a and b at p1
            # and some point between b and c at p2
            # so the space specified to traverse is 1-p1 + p2

            space_in_between_seeds = 1 - self.i_seed_start.p + self.i_seed_end.p
            start_seed_space_per = (1 - self.i_seed_start.p) / space_in_between_seeds
            end_seed_space_per = self.i_seed_end.p / space_in_between_seeds

            inflection_point_per = start_seed_space_per

            if time_into_section_per < inflection_point_per:
                per_into_start_seed_per = time_into_section_per / inflection_point_per
                amnt_in_start_seed = 1 - self.i_seed_start.p
                eval_p = self.i_seed_start.p + (amnt_in_start_seed * per_into_start_seed_per)
                ret = self.i_seed_start(eval_p)
            else:
                time_per_remaining = (1 - inflection_point_per)
                if time_per_remaining == 0:
                    return self.i_seed_end
                per_into_end_seed_per = (time_into_section_per - inflection_point_per) / (1 - inflection_point_per)
                amnt_in_end_seed = self.i_seed_end.p
                eval_p = (amnt_in_end_seed * per_into_end_seed_per)
                ret = self.i_seed_end(eval_p)

            return ret

        def __repr__(self):
            return f"InterpolatableSection({self.i_seed_start}, {self.i_seed_end}, {self.i_time})"

        def __str__(self):
            return f"InterpolatableSection({self.i_seed_start}, {self.i_seed_end}, {self.i_time})"

        def __contains__(self, item):
            return self.i_time.__contains__(item)

    def __init__(self, seeds: List[int] or List[InteropSeed], times: List[float] or List[TimeSection]):
        if isinstance(seeds[0], int) and isinstance(times[0], (float, int)):
            self.seeds = seeds
            self.times = times
            if len(seeds) != len(times):
                raise ValueError("seeds and times must be the same length")

            # Check for negative time values
            if any(t < 0 for t in times):
                raise ValueError("Negative time values are not allowed")

            # create interop seeds
            self.interop_seeds = []
            for i in range(len(seeds) - 1):
                self.interop_seeds.append(StoryBoardSeed.InteropSeed(seeds[i], seeds[i + 1], 0))
                self.interop_seeds.append(StoryBoardSeed.InteropSeed(seeds[i], seeds[i + 1], 1))

            # filter out seeds that are the same
            for i in range(len(seeds) - 1):
                if self.interop_seeds[i].y == self.interop_seeds[i + 1].x:
                    if (self.interop_seeds[i].p == 1) and (self.interop_seeds[i + 1].p == 0):
                        self.interop_seeds.pop(i)

            # create the time sections
            self.time_sections = []
            for i in range(len(times) - 1):
                self.time_sections.append(StoryBoardSeed.TimeSection(times[i], times[i + 1]))

                # Check for overlapping time sections
                if self.time_sections[-1].x >= self.time_sections[-1].y:
                    raise ValueError("Overlapping time sections are not allowed")

            # create the interpolatable sections
            self.interpolatable_sections = []
            for i in range(len(self.interop_seeds) - 1):
                self.interpolatable_sections.append(
                    StoryBoardSeed.InterpolatableSection(self.interop_seeds[i], self.interop_seeds[i + 1],
                                                         self.time_sections[i])
                )
            pass
        elif isinstance(seeds[0], Interpolatable) and isinstance(times[0], Interpolatable):
            self.interop_seeds = seeds
            self.time_sections = times
            if len(seeds) == len(times) + 1:
                # create the interpolatable sections
                self.interpolatable_sections = []
                for i in range(len(self.interop_seeds) - 1):
                    self.interpolatable_sections.append(
                        StoryBoardSeed.InterpolatableSection(self.interop_seeds[i], self.interop_seeds[i + 1],
                                                             self.time_sections[i])
                    )
            self.times = {tm.x for tm in self.time_sections}
            pass
        else:
            raise ValueError("seeds and times must be of the same type")

    def get_time_points(self):
        """ This function returns the time points that the seed changes at """
        return self.times

    def get_seed_at_time(self, seconds) -> InteropSeed:
        """
        >>> times = [0,4,8]
        >>> seeds = [1,2,3]
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(4.01)
        InteropSeed(2, 3, 0.004999999999999893)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(4.0)
        InteropSeed(2, 3, 0)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(0.0)
        InteropSeed(1, 2, 0.0)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(1.0)
        InteropSeed(1, 2, 0.25)
        """

        # find the section
        for section in self.interpolatable_sections:
            if seconds in section:
                # found the correct section
                break
        else:
            raise ValueError("seconds is not in any section")

        # get the seed and time interpolatables
        intr_seed = section.get_i_seed(seconds)
        return intr_seed
        intr_time = section.i_time

        # get the start and end seeds
        start_seed = intr_seed.x
        end_seed = intr_seed.y

        # get the start and end times
        start_time = intr_time.x
        end_time = intr_time.y

        # get the time range
        time_range = end_time - start_time

        # get the time progress
        time_progress = seconds - start_time

        # get the time progress percent
        time_progress_percent = time_progress / time_range

        # this happens when seconds == end_time
        if time_progress_percent == 1.0:
            time_progress_percent = 0.0
            start_seed = end_seed
            end_seed = None

        return Interpolatable(start_seed, end_seed, time_progress_percent)

    def get_seeds_at_times(self, seconds_list):
        """
        >>> seeds = [1, 2, 3]
        >>> times = [0, 4, 8]
        >>> sbseeds = StoryBoardSeed(seeds, times)
        >>> sbseeds.get_seeds_at_times([0.0, 1.0, 4.0, 4.01])
        [InteropSeed(1, 2, 0.0), InteropSeed(1, 2, 0.25), InteropSeed(2, 3, 0), InteropSeed(2, 3, 0.004999999999999893)]

        # If the get_prime_seeds_at_times method is implemented, include the doctest for it in the appropriate method
        # >>> sbseeds.get_prime_seeds_at_times([0.0, 1.0, 4.0, 4.01])
        # [1, 1, 1, 2]
        # >>> sbseeds.get_prime_seeds_at_times(0)
        # [1]
        """
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s) for s in seconds_list]

    def get_prime_seeds_at_times(self, seconds_list):
        """
        >>> seeds = [1, 2, 3]
        >>> times = [0, 4, 8]
        >>> sbseeds = StoryBoardSeed(seeds, times)

        # Test with a single float value
        >>> sbseeds.get_prime_seeds_at_times(0.0)
        [1]

        # Test with a single integer value
        >>> sbseeds.get_prime_seeds_at_times(1)
        [1]

        # Test with a list of float and integer values
        >>> sbseeds.get_prime_seeds_at_times([0.0, 1.0, 4.0, 4.01])
        [1, 1, 2, 2]
        """
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s).x for s in seconds_list]

    def get_subseeds_at_times(self, seconds_list):
        """
        >>> seeds = [1, 2, 3]
        >>> times = [0, 4, 8]
        >>> sbseeds = StoryBoardSeed(seeds, times)

        # Test with a single float value
        >>> sbseeds.get_subseeds_at_times(0.0)
        [2]

        # Test with a single integer value
        >>> sbseeds.get_subseeds_at_times(1)
        [2]

        # Test with a list of float and integer values
        >>> sbseeds.get_subseeds_at_times([0.0, 1.0, 4.0, 4.01])
        [2, 2, 3, 3]
        """
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s).y for s in seconds_list]

    def get_subseed_strength_at_times(self, seconds_list):
        """
           >>> seeds = [1, 2, 3]
           >>> times = [0, 4, 8]
           >>> sbseeds = StoryBoardSeed(seeds, times)

           # Test with a single float value
           >>> sbseeds.get_subseed_strength_at_times(0.0)
           [0.0]

           # Test with a single integer value
           >>> sbseeds.get_subseed_strength_at_times(1)
           [0.25]

           # Test with a list of float and integer values
           >>> sbseeds.get_subseed_strength_at_times([0.0, 1.0, 4.0, 4.01])
           [0.0, 0.25, 0, 0.004999999999999893]
           """
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s).p for s in seconds_list]

    def __repr__(self):
        return f"StoryBoardSeed(seeds={self.interop_seeds}, times={self.time_sections})"

    def __getitem__(self, item):
        """
        >>> test= StoryBoardSeed(
        ... seeds = [1,2,3],times = [0., 4., 8.]
        ... )
        >>> test[2:8]
        StoryBoardSeed(seeds=[InteropSeed(1, 2, 0.5), InteropSeed(2, 3, 0), InteropSeed(2, 3, 1.0)], times=[TimeSection(2.0, 4.0), TimeSection(4.0, 8.0)])
        >>> test= StoryBoardSeed(
        ... seeds = [1,2,3],times = [0., 4., 8.]
        ... )
        >>> test[2:7.9]
        StoryBoardSeed(seeds=[InteropSeed(1, 2, 0.5), InteropSeed(2, 3, 0), InteropSeed(2, 3, 0.9500000000000002)], times=[TimeSection(2.0, 4.0), TimeSection(4.0, 7.9)])



        """
        if isinstance(item, slice):
            start_time = float(item.start)
            stop_time = float(item.stop)
            step = item.step
            if start_time is None:
                start_time = 0
            if stop_time is None:
                stop_time = self.interpolatable_sections[-1].i_time.y
            if step is not None:
                raise ValueError("step is not supported")

            # so really we need to evaluate the object at start, stop and every time index
            # in sellf._times that resides in between start and stop
            new_seeds = UniqueList()
            new_times = UniqueList()
            new_sections = []
            sections_to_evaluate = []
            for section in self.interpolatable_sections:
                if start_time in section:
                    sections_to_evaluate.append(section)

            total_slice_intr = StoryBoardSeed.TimeSection(start_time, stop_time, 0.0)

            for section in self.interpolatable_sections[:-1]:
                if section.i_time in total_slice_intr:
                    sections_to_evaluate.append(copy.deepcopy(section))

            for section in self.interpolatable_sections:
                if stop_time in section:
                    sections_to_evaluate.append(copy.deepcopy(section))

            for section in sections_to_evaluate:
                new_times.append(copy.deepcopy(section.i_time))

            new_times[0].x = start_time
            new_times[0].y = new_times[1].x

            new_times[-1].y = stop_time
            new_times[-1].x = new_times[-2].y

            for tm in new_times:
                seedx = self.get_seed_at_time(tm.x)
                seedy = self.get_seed_at_time(tm.y)
                new_seeds.append(seedx)
                new_seeds.append(seedy)
            new_seeds[0] = self.get_seed_at_time(start_time)
            new_seeds[-1] = self.get_seed_at_time(stop_time)

            new_obj = StoryBoardSeed(new_seeds, new_times)

            return new_obj

        return self.get_seed_at_time(item)

    def _test_interpolatable_section(self):
        """
        >>> i_seed_start = StoryBoardSeed.InteropSeed(1, 2, 0)
        >>> i_seed_end = StoryBoardSeed.InteropSeed(2, 3, 1)
        >>> i_time = StoryBoardSeed.TimeSection(0, 1)
        >>> ints = StoryBoardSeed.InterpolatableSection(i_seed_start, i_seed_end, i_time)
        >>> ints.get_i_seed(0.5)
        InteropSeed(2, 3, 0.0)
        >>> ints.get_i_seed(0.25)
        InteropSeed(1, 2, 0.5)
        >>> ints.get_i_seed(0.75)
        InteropSeed(2, 3, 0.5)
        """
        pass


class StoryBoardData:
    def __init__(self, storyboard_prompt: StoryBoardPrompt, storyboard_seed: StoryBoardSeed):
        self.storyboard_prompt = storyboard_prompt
        self.storyboard_seed = storyboard_seed


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    exit()
    print(StoryBoardPrompt(
        [
            "super:1 hero:1 cat:0.0",
            "super:0 hero:1 cat:0.5",
            "super:1 hero:1 cat:0.0"
        ],
        seconds_lengths=[1.0, 1.0])(0.5))
    assert StoryBoardPrompt(
        [
            "super:1 hero:1 cat:0.0",
            "super:0 hero:1 cat:0.5",
            "super:1 hero:1 cat:0.0"
        ],
        seconds_lengths=[1.0, 1.0])(0.5)[0] == '(super:0.50000000)(hero:1.00000000)(cat:0.25000000)'

    test1 = StoryBoardPrompt("doctests", [0.5, 0.5])
    doctest.testmod()
    doctest.testmod(verbose=True)

    # usage example
    movie_section_lengths = [10, 10]
    movies_fps = 1
    SBP = StoryBoardPrompt(["super:0 hero:0 cat:0.0 anime:.5",
                            "super:1 hero:1 cat:1.0 anime:.5",
                            "super:1 hero:1 cat:0.5 anime:.75",
                            ], movie_section_lengths)
    movie_length_in_frames = sum(movie_section_lengths) * movies_fps
    movie_seconds_per_frame = 1 / movies_fps
    movie_frames = [SBP(i * movie_seconds_per_frame) for i in range(movie_length_in_frames)]
    print(movie_frames, len(movie_frames))
