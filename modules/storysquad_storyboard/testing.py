from modules.storysquad_storyboard.storyboard import SBIHyperParams


def get_test_storyboard(prompt_array: list[str]=None):
    if not prompt_array:
        test_data = [
            "(dog:1) cat:0",
            "(dog:1) cat:1",
            "(dog:0) cat:1",
        ]
    else:
        test_data = prompt_array
    storyboard_params = [SBIHyperParams(
        prompt=prompt,
        seed=i,
        negative_prompt="",
        cfg_scale=7,
        steps=5,
        subseed=-1,
        subseed_strength=0.0,
    ) for i, prompt in enumerate(test_data)]
    out = storyboard_params[0]
    for sbp in storyboard_params[1:]:
        out = out + sbp
    return out