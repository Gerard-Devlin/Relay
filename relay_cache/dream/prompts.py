"""DREAM benchmark formatting shared by every compared method."""

def prompt_ids(tokenizer, text):
    if not isinstance(text, str) or not text:
        raise ValueError('Nonempty legitimate prompt required')
    if not isinstance(tokenizer.bos_token, str):
        raise ValueError('DREAM official add_bos_token profile requires a BOS token')
    # Match official DREAM eval scripts: BOS + already prepared few-shot prompt.
    # Do not apply LLaDA's chat template or append reference/test information.
    return tokenizer(tokenizer.bos_token + text)['input_ids']
