"""Explicit DREAM formatting; official baseline completion prompts stay intact."""

CHAT_POLICY = 'DREAM Instruct chat template; prepared legitimate prompt as user content'

def prompt_ids(tokenizer, text, *, style='completion'):
    if not isinstance(text, str) or not text:
        raise ValueError('Nonempty legitimate prompt required')
    if style == 'chat':
        return tokenizer.apply_chat_template([{'role':'user','content':text}],
            tokenize=True, add_generation_prompt=True)
    if style != 'completion':
        raise ValueError('Unknown DREAM prompt policy')
    if not isinstance(tokenizer.bos_token, str):
        raise ValueError('DREAM official add_bos_token profile requires a BOS token')
    # Match official DREAM eval scripts: BOS + already prepared few-shot prompt.
    # Do not apply LLaDA's chat template or append reference/test information.
    prepared = text if text.startswith(tokenizer.bos_token) else tokenizer.bos_token + text
    return tokenizer(prepared)['input_ids']
