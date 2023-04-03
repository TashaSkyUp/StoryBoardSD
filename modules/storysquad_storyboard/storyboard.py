import json
import numpy as np
import os
import numbers
from dataclasses import dataclass
from typing import List, Any
import matplotlib.pyplot as plt
import spacy
import re

DEV_MODE = os.getenv("STORYBOARD_DEV_MODE", "False") == "True"
DEFAULT_HYPER_PARAMS = {
    "prompt": "",
    "negative_prompt": "",
    "steps": 6,
    "seed": -1,
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

if DEV_MODE:
    DEFAULT_HYPER_PARAMS = DEV_HYPER_PARAMS
else:
    DEFAULT_HYPER_PARAMS = DEV_HYPER_PARAMS


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
        return SBIHyperParams(prompt=self._prompt[item], negative_prompt=self.negative_prompt[item],
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
    if prompt == "":
        raise ValueError("prompt cannot be empty")
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
    prompt = prompt.replace('-', "")

    # compact blankspace
    for i in range(10):
        prompt = prompt.replace("  ", " ")

    prompt = prompt.replace("(", "").replace(")", "")
    return prompt.strip()


def remove_all_but_words(prompt):
    """
        >>> remove_all_but_words("Get :1.2 busy:1. living:0.4 or:0 get: 0 busy:0 dying:0.6.")
        'Get busy living or get busy dying'
        >>> remove_all_but_words(["Get :1.2 busy:1. living:0.4 or:0 get: 0 busy:0 dying:0.6.","Get :1.2 busy:1. living:0.4 or:0 get: 0 busy:0 dying:0.6."])
        ['Get busy living or get busy dying', 'Get busy living or get busy dying']
        """
    if isinstance(prompt, list):
        res = []
        for p in prompt:
            res.append(remove_all_but_words(p))
    else:
        chars = re.compile('[^a-zA-Z\s]+')
        clean = chars.sub(' ', prompt)
        res = ' '.join(clean.split()).strip()

    return res


from typing import List, Tuple


def _get_noun_list() -> List[str]:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("nounlist.csv", 'r') as f:
        noun_list = f.read().splitlines()
    return noun_list


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

    >>> try:
    ...     SB = StoryBoardPrompt("doctests", [.5,.5])
    ...     SB._get_prompt_at_time(0.5)
    ... except Exception as e:
    ...     raise(e)
    '(dog:1.00000000)(cat:1.00000000)'
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
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_prompts =[self.nlp(remove_all_but_words(p)) for p in self._sanitized_prompts]
        self.prompt_pos_dicts = self.get_all_pos()  # Creates a dict to reference each POS for current prompt

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
            else:
                raise TypeError("args must be a float or a list of floats")
        except IndexError:
            raise TypeError("args must be a float or a list of floats")

    def get_all_pos(self) -> dict:
        """
        Given a prompt, returns a dictionary containing the POS of all the words in the prompt.

        Args:
        prompt (str): The prompt to be processed.

        Returns:
        dict: A dictionary containing the words and their corresponding POS.

        Examples:
        >>> SB = StoryBoardPrompt("doctests", [.5,.5])
        >>> SB.get_all_pos()
        [{'dog': 'NOUN', 'cat': 'NOUN'}, {'dog': 'NOUN', 'cat': 'NOUN'}, {'dog': 'NOUN', 'cat': 'NOUN'}]

        >>> SB.get_all_pos()
        [{'dog': 'NOUN', 'cat': 'NOUN'}, {'dog': 'NOUN', 'cat': 'NOUN'}, {'dog': 'NOUN', 'cat': 'NOUN'}]
        """
        outlist = []

        for item in self.nlp_prompts:
            pos_dict = {}
            for token in item:
                pos_dict[str(token)] = token.pos_ #creates a Key : Element pair of each word in the prompt and it's pos
            outlist.append(pos_dict)
        return outlist

    def get_word_pos(self, word) -> str:
        """
        Given a word, returns its POS.

        Args:
        word (str): The word to be processed.

        Returns:
        str: The POS of the given word.

        Examples:
        >>> SB = StoryBoardPrompt("doctests", [.5,.5])
        >>> SB.get_word_pos("cat")
        'NOUN'
        >>> SB.get_word_pos("dog")
        'NOUN'

        """

        if not word or not isinstance(word, str) or not word[0].isalpha():
            return None

        # get word's POS from the dictionary, or return None if the word is not present
        # uses the first dict which for now is the same as all others
        pos = self.prompt_pos_dicts[0].get(word, None)
        return pos
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
        prompt = prompt.replace('-', "")
        # compact blankspace
        for i in range(10):
            prompt = prompt.replace("  ", " ")

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
        ...     print(e.message)
        [[[('dog', 1.0), ('cat', 0.0)], [('dog', 1.0), ('cat', 1.0)]], [[('dog', 1.0), ('cat', 1.0)], [('dog', 0.0), ('cat', 1.0)]]]
        """
        sections: List[List[List[Tuple[str, float]]]] = [
            [words_and_weights_list[0], words_and_weights_list[1]],
            # transition from lattice pt 1 to 2
            [words_and_weights_list[1], words_and_weights_list[2]],
            # transition from lattice pt 2 to 3
        ]
        return sections



    def _get_word_weight_at_percent(self, section, word_index, percent):
        """
        Calculate the weight of a word based on its section traversal progress.

        :param section: list of two tuples containing the weight values of a word at the beginning and end of a section
        :param word_index: integer representing the index of the word whose weight is being calculated
        :param percent: float representing the percentage of the section that has been traversed
        :return: float representing the weight of the word


        >>> test_obj = StoryBoardPrompt("doctests", [.5,.5])
        >>> percent_values = np.linspace(0, 1, 30)
        >>> weights = [[test_obj._get_word_weight_at_percent(test_obj._sections[0], word_index=i, percent=p) for p in percent_values] for i in range(2)]
        >>> plt.plot(percent_values, weights[0])
        >>> plt.plot(percent_values, weights[1])
        >>> plt.xlabel('Percent')
        >>> plt.ylabel('Weight')
        >>> plt.title('Word weight over section traversal')
        >>> plt.show()

       """
        curr_word = section[0][word_index][0]  # word text
        action_list = ['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']

        pos = self.get_word_pos(curr_word)
        # Compute the transition weight as a linear interpolation between the start and end weights
        start_weight = section[0][word_index][1]  # word weight
        end_weight = section[1][word_index][1]
        linear_weight = start_weight + percent * (end_weight - start_weight)
        if pos in action_list:
            # Compute the cosinusoidal weights as a function of linear weight
            frequency: float = 3
            amplitude: float = -1 / 10  # -1 to 1
            # Compute the sinusoidal weights as a function of linear weight y=(sin(X*pi*10*2)+1)/2 where X== linear weight
            sinusoidal_weight =  (np.sin(2 * np.pi * frequency * amplitude * linear_weight)+1)/2
            cosinusoidal_weight = (np.cos(2 * np.pi * percent * frequency) * amplitude) + linear_weight + (abs(amplitude))
            if cosinusoidal_weight > 1.5:
                cosinusoidal_weight -= 0.5
            return min(cosinusoidal_weight, 1)
        return (linear_weight * 0.8) #assigns penalty to weight of words not represented by 'action_list'


    def _get_frame_values_for_prompt_word_weights(self, sections,
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
                    word_frame_weight = self._get_word_weight_at_percent(section, word_idx,
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
        '(dog:0.70000000)(cat:1.20000000)'

        """

        if seconds < 0:
            raise ValueError("seconds cannot be less than 0")
        if seconds > self.total_seconds:
            raise ValueError("seconds cannot be greater than the total time of the storyboard")

        # find the section that the seconds is in using self._times_sections_start
        section_second_is_in = len(self._times_sections_start) - 1
        for i, section_start_time in enumerate(self._times_sections_start):
            if seconds < section_start_time:
                section_second_is_in = i - 1
                break

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

    def __getitem__(self, time_seconds: float) -> str:
        return self(time_seconds)

    def _get_nouns_only(self, p):
        out = [sp for sp in p if sp[0] in self.noun_list]
        return out


class StoryBoardSeed():

    def __init__(self, seeds: [int], times: [int]):
        self.seeds = seeds
        self.times = times

    def get_seed_at_time(self, seconds):
        """
        >>> times = [4,8,16]
        >>> seeds = [1,2,3]
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(0.0)
        (1, 2, 0.0)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(1.0)
        (1, 2, 0.25)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(4.0)
        (1, 2, 1.0)
        >>> StoryBoardSeed(seeds,times).get_seed_at_time(4.01)
        (2, 3, 0.0024999999999999467)


        """
        times = self.times
        times = [0] + times
        seeds = self.seeds
        import numpy as np
        if seconds < times[0]:
            raise ValueError("seconds cannot be less than the start time of the storyboard")
        if seconds > times[-1]:
            raise ValueError("seconds cannot be greater than the end time of the storyboard")
        index = np.searchsorted(times, seconds) - 1
        index = max(index, 0)
        total_time_in_section = times[index] - times[index + 1]
        time_into_section = seconds - times[index]
        percent_into_section = time_into_section / total_time_in_section
        return seeds[index], seeds[index + 1], abs(percent_into_section)

    def get_seeds_at_times(self, seconds_list):
        """
        >>> times = [4,8,16]
        >>> seeds = [1,2,3]
        >>> sbseeds = StoryBoardSeed(seeds,times)
        >>> sbseeds.get_seeds_at_times([0.0,1.0,4.0,4.01])
        [(1, 2, 0.0), (1, 2, 0.25), (1, 2, 1.0), (2, 3, 0.0024999999999999467)]
        >>> sbseeds.get_prime_seeds_at_times([0.0,1.0,4.0,4.01])
        [1, 1, 1, 2]
        >>> sbseeds.get_prime_seeds_at_times(0)
        [1]
        """
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s) for s in seconds_list]

    def get_prime_seeds_at_times(self, seconds_list):
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s)[0] for s in seconds_list]

    def get_subseeds_at_times(self, seconds_list):
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s)[1] for s in seconds_list]

    def get_subseed_strength_at_times(self, seconds_list):
        if isinstance(seconds_list, float) or isinstance(seconds_list, int):
            seconds_list = [seconds_list]
        return [self.get_seed_at_time(s)[2] for s in seconds_list]


class StoryBoardData:
    def __init__(self, storyboard_prompt: StoryBoardPrompt, storyboard_seed: StoryBoardSeed):
        self.storyboard_prompt = storyboard_prompt
        self.storyboard_seed = storyboard_seed


if __name__ == "__main__":
    import modules.paths
    import doctest

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
