import multiprocessing
from multiprocessing import Pool, Process, Queue
import re
from collections import namedtuple
from functools import lru_cache
from typing import List
import lark
import time

srvcs = []
srvcs_queue_x = None
srvcs_queue_y = None
long_test_prompt = """"(Once:1.1614936970214988) (upon:1.0770700732517209) (a:0.8057376874146277) (
time:1.4315311440182081) (there:0.7807131569489789) (was:0.8749109313791749) (a:0.9072029956888351) (
brother:1.112638178740016) (and:1.299332187139783) (sister:0.554181364401428) (named:0.7239363960988605) (
Jack:1.4419724712954172) (and:0.882282461313189) (Jill:0.7005698833995918) (They:1.3080224491269612) (
lived:1.329565170316422) (in:0.5320844582236661) (a:0.9393414219583508) (small:0.624843334585973) (
house:1.3229849286706927) (with:0.5493283147697796) (their:1.0584556348990204) (pet:0.9978727038657568) (
dog:0.8332579225360421) (Rover:0.9622579229003417) (One:0.8028191769520655) (day:0.9662454013951979) (
while:0.7914420401290833) (playing:0.9654057820035488) (in:1.4339765055890095) (their:1.204670788079858) (
backyard:0.9973976959292874) (they:1.2769500224130834) (found:0.506427000226051) (a:1.2379524306938374) (
box:1.4394389261757499) (of:1.233112108661976) (old:0.9706050015319738) (spaceship:1.4851627108397953) (
parts:0.6446621524600065) (They:0.9837736431514666) (decided:1.2522870554068755) (to:1.4383764396269867) (
build:0.6693394331866983) (a:0.9072290102451693) (spaceship:1.2552849076637438) (and:0.6003889990933388) (
go:1.1419735863316616) (on:0.7073903475600017) (an:1.2015480930932827) (adventure:1.2299094302608597) (
to:0.912734154081717) (the:1.267219322734948) (moon.:1.03880111432691) (Rover:0.5232622994692313) (
helped:1.045592625400766) (them:1.4342054479814959) (gather:0.9963277536213888) (all:1.0813148254281004) (
the:1.250279954872888) (necessary:1.220434933472268) (materials:0.747241226130175) (and:0.6515032405110619) (
they:1.44559098918824) (worked:1.0576859746750351) (hard:1.018874208823275) (to:0.8202937285507461) (
build:1.1761828726000538) (the:1.1018568032739555) (spaceship:0.7543205753412717) (After:0.5838055021490325) (
many:1.1754486736885594) (days:1.1652219868920404) (of:0.5432137617104827) (hard:1.1872212700269853) (
work:0.5622336918838317) (their:1.4272530034472486) (spaceship:1.1426361211346592) (was:0.5654190801710435) (
finally:1.4779542278540665) (ready:0.5628731586367645) (They:1.3726426170199204) (climbed:1.029730816819138) (
aboard:1.4209890985920728) (and:0.9473404846584957) (blasted:1.0609072516906553) (off:0.6792022269612792) (
into:0.5521516712189346) (space:0.9599065014779821) (They:1.2861677831902496) (flew:0.7073944578611968) (
to:1.4754103986929163) (the:1.2089875038068574) (moon:0.5791709087155436) (and:0.8816436040570048) (
had:0.8916283618572423) (many:0.6797482079497513) (exciting:1.4501006874334528) (adventures:1.2313536456330532) (
but:1.35870587924907) (eventually:1.0806198308267168) (it:1.3916554098015572) (was:0.7240111057738385) (
time:1.062523618878417) (to:1.1246257230123529) (return:0.9850349304289857) (to:1.335494904544591) (
Earth:0.5629684464111863) (They:1.138162271105184) (safely:0.9939707999541673) (landed:1.488355473376202) (
back:0.5872966922684766) (in:0.9237086842528535) (their:0.7521417678564594) (backyard:0.7155210427088046) (
and:0.9163922281380649) (couldn't:1.152940426529708) (wait:0.5785318933823775) (to:1.0616977492138582) (
tell:0.6275546585886305) (their:0.7999445923488155) (friends:1.2068174334458708) (and:1.0530124721244718) (
family:0.7436772303874467) (about:1.4967143358852113) (their:1.1258744428862226) (amazing:1.1588294422940584) (
trip:0.8796421607704908) (to:1.3227901118046828) (the:1.242386866983527) (moon.:1.4621010282066895) """

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def get_learned_conditioning_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    """

def collect_steps(steps, tree):
    l = [steps]

    class CollectSteps(lark.Visitor):
        def scheduled(self, tree):
            tree.children[-1] = float(tree.children[-1])
            if tree.children[-1] < 1:
                tree.children[-1] *= steps
            tree.children[-1] = min(steps, int(tree.children[-1]))
            l.append(tree.children[-1])

        def alternate(self, tree):
            l.extend(range(1, steps + 1))

    CollectSteps().visit(tree)
    return sorted(set(l))


def at_step(step, tree):
    class AtStep(lark.Transformer):
        def scheduled(self, args):
            before, after, _, when = args
            yield before or () if step <= when else after

        def alternate(self, args):
            yield next(args[(step - 1) % len(args)])

        def start(self, args):
            def flatten(x):
                if type(x) == str:
                    yield x
                else:
                    for gen in x:
                        yield from flatten(gen)

            return ''.join(flatten(args))

        def plain(self, args):
            yield args[0].value

        def __default__(self, data, children, meta):
            for child in children:
                yield child

    return AtStep().transform(tree)


def get_schedule(prompt, steps):
    try:
        ttt = time.time()
        tree = schedule_parser.parse(prompt)
        print(f'parse time: {time.time() - ttt}')
    except lark.exceptions.LarkError as e:
        if 0:
            import traceback
            traceback.print_exc()
        return [[steps, prompt]]
    return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]


def get_schedule_dict(prompt, steps):
    o = get_schedule(prompt, steps)
    # {prompt: get_schedule(prompt, steps) for prompt in set(prompts)}
    return [prompt, o]


def get_learned_conditioning_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    """

    # TODO: Multi processing for this
    # create a Pool with 8 processes
    # with modules.shared.mp_pool() as pool:
    #    # process prompts iterable with pool
    #    ret = pool.starmap(get_schedule_dict, zip(set(prompts), [steps] * len(prompts)))
    #    ret2 = {i[0]: i[1] for i in ret}
    # promptdict = ret2
    promptdict = {prompt: get_schedule(prompt, steps) for prompt in set(prompts)}

    return [promptdict[prompt] for prompt in prompts]

def srvc_prompt_schedule(x_q:multiprocessing.Queue,y_q):
    y_q.put("ready")
    while True:
        try:
            d = x_q.get()
            #print(f'srvcs received {d}')
            if len (d) != 2:
                break
            prompt, steps = d
            #res2= get_schedule(prompt, steps)
            res1=get_schedule_dict(prompt, steps)
            y_q.put(res1)
            print(f'srvcs sent one')
        except multiprocessing.queues.Empty as e:
            print(e)
            continue

def get_learned_conditioning_prompt_schedules_mp_process(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    """
    global srvcs, srvcs_queue_x, srvcs_queue_y
    if srvcs ==[]:
        print(f"starting srvcs for prompt parser")
        srvcs_queue_x = multiprocessing.Queue()
        srvcs_queue_y = multiprocessing.Queue()
        for i in range(12):
            srvcs.append(Process(target=srvc_prompt_schedule,
                                 args=(srvcs_queue_x,srvcs_queue_y),
                                 name=f"srvcs[{i}]"
                                 ))
            srvcs[-1].start()
        for i in range(12):
            print(f'srvc thread {i} is {srvcs_queue_y.get()}')


    prompts: List[str] = prompts
    set_prompts = set(prompts)

    steps: int = steps
    steps_list = [steps] * len(prompts)


    #queue_list = [srvcs_queue] * len(prompts)

    star_args = zip(set_prompts, steps_list)
    star_args_list = list(star_args)
    for i,p in enumerate(star_args_list):
        print(f'putting {i} of {len(star_args_list)}')
        srvcs_queue_x.put(obj=p)

    ret = []
    for _ in range(len(star_args_list)):
        print(f'waiting for {len(star_args_list) - len(ret)}')
        ret.append(srvcs_queue_y.get())
        print(f'got {len(star_args_list) - len(ret)}')

    ret2 = {i[0]: i[1] for i in ret}
    promptdict = ret2

    return [promptdict[prompt] for prompt in prompts]

    star_args_list = list(star_args)
    ttt=time.time()
    procs = [Process(target=get_schedule_dict_to_queue, args=args) for args in star_args_list]
    for p in procs:
        p.start()
    res = [srvcs_queue.get() for _ in range(len(star_args_list))]
    for p in procs:
        p.join()
    promptdict = {i[0]: i[1] for i in res}
    print(f'get_learned_conditioning_prompt_schedules_mp_process time: {time.time() - ttt}')
    return [promptdict[prompt] for prompt in prompts]


def get_learned_conditioning_prompt_schedules_mp_pool(prompts, steps, pool):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    """
    if pool is None:
        pool = multiprocessing.Pool(16)
    with pool:
        # process prompts iterable with pool
        prompts: List[str] = prompts
        set_prompts = set(prompts)

        steps: int = steps
        steps_list = [steps] * len(prompts)

        star_args = zip(set_prompts, steps_list)
        star_args_list = list(star_args)
        # TODO: evaluate by hand to see if the mp version has the same results as the non-mp version
        ret = pool.starmap(get_schedule_dict, star_args_list)
        ret2 = {i[0]: i[1] for i in ret}
    promptdict = ret2

    return [promptdict[prompt] for prompt in prompts]


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


def get_learned_conditioning(model, prompts: tuple, steps, pool=None):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    prompts = list(prompts)
    res = []
    skip_lc_p_schedules = True
    if not skip_lc_p_schedules:
        prompt_schedules = get_learned_conditioning_prompt_schedules_mp_process(prompts, steps)
    else:
        prompt_schedules = [[[steps,prompt]] for prompt in prompts]

    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^(.*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


@lru_cache(maxsize=20)
def get_multicond_prompt_list(prompts: tuple):
    res_indexes = []

    prompt_flat_list = []
    prompt_indexes = {}

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: List[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: List[List[ComposableScheduledPromptConditioning]] = batch


@lru_cache(maxsize=20)
def get_multicond_learned_conditioning(model, prompts: tuple, steps, debug=False) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """
    # prompts=list(prompts)
    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)
    learned_conditioning = get_learned_conditioning(model, tuple(prompt_flat_list), steps)
    res = []

    if debug: print(f'starting inner loop for get_multicond')
    if debug: ttt = time.time()

    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])
    if debug:  print(f'finished inner loop for get_multicond in {time.time() - ttt} seconds')
    ret = MulticondLearnedConditioning(shape=(len(prompts),), batch=res)
    return ret


def reconstruct_cond_batch(c: List[List[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)
    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = current
                break
        res[i] = cond_schedule[target_index].cond

    return res


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for batch_no, composable_prompts in enumerate(c.batch):
        conds_for_batch = []

        for cond_index, composable_prompt in enumerate(composable_prompts):
            target_index = 0
            for current, (end_at, cond) in enumerate(composable_prompt.schedules):
                if current_step <= end_at:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    # if prompts have wildly different lengths above the limit we'll get tensors fo different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    return conds_list, torch.stack(tensors).to(device=param.device, dtype=param.dtype)


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0], ['house', 1.5730000000000004], [' ', 1.1], ['on', 1.0], [' a ', 1.1], ['hill', 0.55], [', sun, ', 1.1], ['sky', 1.4641000000000006], ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:
    import torch  # doctest faster
