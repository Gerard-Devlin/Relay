# Official zero-shot query formatters from LLaDA-V evaluation, Apache-2.0; see NOTICE.
def mathvista_query(self, problem, shot_type, examples=None, shot_num=0, use_caption=False, use_ocr=False):
    if shot_num == 0:
        demo_prompt = ''
    else:
        demos = []
        shot_num = min(shot_num, len(examples))
        for example in examples[:shot_num]:
            prompt = ''
            prompt += f"Question: {example['question']}"
            if 'choices' in example:
                texts = ['Choices:']
                for i, choice in enumerate(example['choices']):
                    texts.append(f"({chr(ord('A') + i)}) {choice}")
                prompt += '\n' + '\n'.join(texts)
            if use_caption:
                caption = example['caption'] if 'caption' in example else ''
                if caption != '':
                    prompt += '\n' + f'Image description: {caption}'
            if use_ocr:
                ocr = example['ocr'] if 'ocr' in example else ''
                if ocr != '':
                    prompt += '\n' + f'Image detected text: {ocr}'
            if shot_type == 'solution':
                solution = example['solution'].strip()
                prompt += '\n' + f'Solution: {solution}'
            if shot_type == 'step-by-step':
                solution = example['solution'].strip()
                prompt += '\n' + f'{solution}'
            if shot_type == 'think-step-by-step':
                solution = example['solution'].strip()
                prompt += '\n' + f'{solution}'
            if shot_type == 'direct':
                solution = example['solution'].strip()
                prompt += '\n' + f'{solution}'
            if shot_type == 'code':
                code = example['code'].strip()
                prompt += '\n' + f'Python code: {code}'
            demos.append(prompt)
        demo_prompt = '\n\n'.join(demos)
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']
    caption = problem['caption']
    ocr = problem['ocr']
    precision = problem['precision']
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    if shot_type == 'solution':
        if question_type == 'multi_choice':
            assert answer_type == 'text'
            hint_text = f'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.'
        else:
            assert answer_type in ['integer', 'float', 'list']
            if answer_type == 'integer':
                hint_text = f'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.'
            elif answer_type == 'float' and precision == 1:
                hint_text = f'Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.'
            elif answer_type == 'float' and precision == 2:
                hint_text = f'Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.'
            elif answer_type == 'list':
                hint_text = f'Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.'
    elif shot_type == 'format-prompt':
        if question_type == 'multi_choice':
            assert answer_type == 'text'
            hint_text = f"Answer with the option's letter from the given choices directly."
        elif answer_type == 'integer':
            hint_text = f'Answer the question using a single integer number.'
        elif answer_type == 'float' and precision == 1:
            hint_text = f'Answer the question using a single floating-point number with one decimal place.'
        elif answer_type == 'float' and precision == 2:
            hint_text = f'Answer the question using a single floating-point number with two decimal places.'
        elif answer_type == 'list':
            hint_text = f'Answer the question using a Python list.'
    elif shot_type == 'step-by-step':
        if question_type == 'multi_choice':
            assert answer_type == 'text'
            hint_text = f'Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.'
        else:
            assert answer_type in ['integer', 'float', 'list']
            if answer_type == 'integer':
                hint_text = f'Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.'
            elif answer_type == 'float' and precision == 1:
                hint_text = f'Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.'
            elif answer_type == 'float' and precision == 2:
                hint_text = f'Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.'
            elif answer_type == 'list':
                hint_text = f'Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.'
    elif shot_type == 'reason-first':
        if question_type == 'multi_choice':
            assert answer_type == 'text'
            hint_text = f'First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.'
        else:
            assert answer_type in ['integer', 'float', 'list']
            if answer_type == 'integer':
                hint_text = f'First perform reasoning, then finally answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end in the following format: Answer: xxx.'
            elif answer_type == 'float' and precision == 1:
                hint_text = f'First perform reasoning, then finally answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end in the following format: Answer: xxx.'
            elif answer_type == 'float' and precision == 2:
                hint_text = f'First perform reasoning, then finally answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end in the following format: Answer: xxx.'
            elif answer_type == 'list':
                hint_text = f'First perform reasoning, then finally answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end in the following format: Answer: xxx.'
    elif shot_type == 'direct':
        hint_text = ''
    else:
        assert shot_type == 'code'
        hint_text = 'Hint: Please generate a python code to solve the problem'
    if shot_type == 'format-prompt':
        question_text = f'{question}'
    else:
        question_text = f'Question: {question}'
    if unit:
        question_text += f' (Unit: {unit})'
    if choices:
        if shot_type == 'format-prompt':
            texts = []
            for i, choice in enumerate(choices):
                texts.append(f"{chr(ord('A') + i)}. {choice}")
            choices_text = '\n'.join(texts)
        else:
            texts = ['Choices:']
            for i, choice in enumerate(choices):
                texts.append(f"({chr(ord('A') + i)}) {choice}")
            choices_text = '\n'.join(texts)
    else:
        choices_text = ''
    caption_text = ''
    if use_caption and caption != '':
        caption_text = f'Image description: {caption}'
    ocr_text = ''
    if use_ocr and ocr != '':
        ocr_text = f'Image detected text: {ocr}'
    if shot_type == 'solution':
        prompt = 'Solution: '
    elif shot_type == 'format-prompt':
        prompt = ''
    elif shot_type == 'step-by-step':
        prompt = ''
    elif shot_type == 'reason-first':
        prompt = ''
    elif shot_type == 'direct':
        prompt = ''
    else:
        assert shot_type == 'code'
        prompt = 'Python code: '
    final_prompt = 'If it is hard to determine, give the closest answer.'
    if shot_type == 'reason-first':
        elements = [question_text, choices_text, caption_text, ocr_text, prompt, hint_text, final_prompt]
        test_query = '\n'.join([e for e in elements if e != ''])
    else:
        elements = [question_text, choices_text, caption_text, ocr_text, hint_text, prompt]
        test_query = '\n'.join([e for e in elements if e != ''])
    query = demo_prompt + '\n\n' + test_query
    query = query.strip()
    return query

def mathverse_query(self, problem, shot_type, hint, query_type, examples=None, shot_num=0, use_caption=False, use_ocr=False):
    if shot_num == 0:
        demo_prompt = ''
    else:
        demos = []
        shot_num = min(shot_num, len(examples))
        for example in examples[:shot_num]:
            prompt = ''
            prompt += f'Question: {example[query_type]}'
            if shot_type == 'solution':
                solution = example['solution'].strip()
                prompt += '\n' + f'Solution: {solution}'
            if shot_type == 'step-by-step':
                solution = example['solution'].strip()
                prompt += '\n' + f'{solution}'
            if shot_type == 'direct':
                solution = example['solution'].strip()
                prompt += '\n' + f'{solution}'
            demos.append(prompt)
        demo_prompt = '\n\n'.join(demos)
    question = problem['question']
    question_type = problem['question_type']
    if shot_type == 'format-prompt':
        hint_text = ''
    elif shot_type == 'custom-prompt':
        if question_type == 'multi-choice':
            hint_text = hint['multi-choice']
        else:
            hint_text = hint['free-form']
    if shot_type == 'format-prompt':
        question_text = f'{problem[query_type]}'
    elif shot_type == 'custom-prompt':
        question_text = f'Question: {question}'
    elements = [hint_text, question_text]
    test_query = '\n'.join([e for e in elements if e != ''])
    query = demo_prompt + '\n\n' + test_query
    query = query.strip()
    return query
