# Adapted from the LLaDA-V MathVista evaluator (MathVista/LLaVA, Apache-2.0); see NOTICE.
import re,logging
def distance(a,b):
    """Unit-cost Levenshtein distance, as in the historical scorer."""
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]
from .visual_templates import DEMO_PROMPT
eval_logger=logging.getLogger(__name__)

class MathVistaEvaluator:

    def __init__(self, api_key, gpt_model='gpt-3.5-turbo', quick_extract=False):
        self.api_key = api_key
        self.gpt_model = gpt_model
        self.quick_extract = quick_extract

    def create_test_prompt(self, demo_prompt, query, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f'{query}\n\n{response}'
        full_prompt = f'{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: '
        return full_prompt

    def extract_answer(self, response, problem, quick_extract=False):
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        choices = problem.get('choices', [])
        query = problem['query']
        if not response:
            return ''
        if question_type == 'multi_choice' and response in choices:
            return response
        if answer_type == 'integer':
            try:
                extraction = int(response)
                return str(extraction)
            except ValueError:
                pass
        if answer_type == 'float':
            try:
                extraction = str(float(response))
                return extraction
            except ValueError:
                pass
        if quick_extract:
            eval_logger.info('Quickly extracting answer...')
            try:
                result = re.search('The answer is "(.*)"\\.', response)
                if result:
                    extraction = result.group(1)
                    return extraction
            except re.error:
                pass
        try:
            full_prompt = self.create_test_prompt(DEMO_PROMPT, query, response)
            extraction = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            return extraction
        except Exception as e:
            eval_logger.error(e)
            eval_logger.error(f'Error in extracting answer for problem')
        return ''

    def get_most_similar(self, prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
        """
        distances = [distance(prediction, choice) for choice in choices]
        ind = distances.index(min(distances))
        return choices[ind]

    def normalize_extracted_answer(self, extraction, choices, question_type, answer_type, precision):
        """
        Normalize the extracted answer to match the answer type
        """
        if question_type == 'multi_choice':
            if isinstance(extraction, str):
                extraction = extraction.strip()
            else:
                try:
                    extraction = str(extraction)
                except:
                    extraction = ''
            letter = re.findall('\\(([a-zA-Z])\\)', extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()
            options = [chr(ord('A') + i) for i in range(len(choices))]
            if extraction in options:
                ind = options.index(extraction)
                extraction = choices[ind]
            else:
                extraction = self.get_most_similar(extraction, choices)
            assert extraction in choices
        elif answer_type == 'integer':
            try:
                extraction = str(int(float(extraction)))
            except:
                extraction = None
        elif answer_type == 'float':
            try:
                extraction = str(round(float(extraction), int(precision)))
            except:
                extraction = None
        elif answer_type == 'list':
            try:
                extraction = str(extraction)
            except:
                extraction = None
        return extraction

    def safe_equal(self, prediction, answer):
        """
        Check if the prediction is equal to the answer, even if they are of different types
        """
        try:
            if str(prediction).strip() == str(answer).strip():
                return True
            return False
        except Exception as e:
            eval_logger.info(e)
            return False
