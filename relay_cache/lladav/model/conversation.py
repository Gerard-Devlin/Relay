# Adapted from LLaDA-V / LLaVA, Apache-2.0; see NOTICE.
import dataclasses
from enum import auto,Enum
from typing import List,Any,Dict,Union,Tuple

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = '###'
    sep2: str = None
    version: str = 'Unknown'
    tokenizer_id: str = ''
    tokenizer: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if 'mmtag' in self.version:
                init_msg = init_msg.replace('<image>', '').strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], '<Image><image></Image>'))
                messages.insert(1, (self.roles[1], 'Received.'))
            elif not init_msg.startswith('<image>'):
                init_msg = init_msg.replace('<image>', '').strip()
                messages[0] = (init_role, '<image>\n' + init_msg)
            else:
                messages[0] = (init_role, init_msg)
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if self.system == '' else self.system + self.sep + '\n'
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = '<image>' * len(images) + message
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            if self.tokenizer is None:
                if self.version == 'llama_v3':
                    self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code=True)
                elif self.version == 'llava_llada':
                    self.tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-V', trust_remote_code=True)
                else:
                    raise ValueError('The tokenizer is not available. Make sure you have the necessary permissions.')
            chat_template_messages = [{'role': 'system', 'content': self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = '<image>' * len(images) + message
                    chat_template_messages.append({'role': role, 'content': message})
            return self.tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ''
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], 'Conversation should alternate user/assistant/user/assistant/...'
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f'<<SYS>>\n{msg}\n<</SYS>>\n\n' if len(msg) > 0 else msg
            wrap_inst = lambda msg: f'[INST] {msg} [/INST]'
            ret = ''
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, 'first message should not be none'
                    assert role == self.roles[0], 'first message should come from user'
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += ' ' + message + ' ' + self.sep2
                else:
                    ret += ''
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ''
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

conv_llava_llada = Conversation(
    system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="llava_llada",
    messages=[],
    offset=0,
    sep="<|eot_id|>",
    sep_style=SeparatorStyle.LLAMA_3,
)

conv_templates={'llava_llada':conv_llava_llada}
