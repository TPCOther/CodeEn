import random
import re
import httpx
from openai import OpenAI
from antlr4 import *
import Comment_models
from typing import List
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.javatokenizer.tokenizer import tokenize_java_code_origin, tokenize_java_code_raw
from utils.utils import getJaccard


def comment_get_zero_shot_prompt(code):

    sys_prompt = f"Assume you have a deep understanding of Java programming language concepts such as loops, conditionals, functions, and data structures.\nGiven a Java function '###Code', generate a short comment in one sentence.\nPut the comment between <Generation> </Generation> tags."

    user_prompt = f"###Code:\n{code.strip()}\n"

    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]


def comment_get_zero_shot_prompt_repair(code,
                                        candidates: List[str],
                                        mode="random"):

    # 初始化的assertion是一个list，需要将其转换为string
    # .
    if mode == "random":
        candidate_list = list(set(candidates))
        # 打乱顺序
        random.shuffle(candidate_list)

    else:
        #去重，但是要保留原始的顺序
        candidate_list = []
        [
            candidate_list.append(x) for x in candidates
            if x not in candidate_list
        ]

    sys_prompt = f"Assume you have a deep understanding of Java programming language concepts such as loops, conditionals, functions, and data structures.\nTake a deep breath and work on the following problem as an experience Java software engineer step by step.\nGiven a Java function '###Code', analyze the function to understand its functionality and then select the most appropriate comment from the given comment list '###Comments' that accurately describes the function's purpose or action.\nPut the chosen comment between <Repair> </Repair> tags."
    user_prompt = f"###Code:\n{code.strip()}\n###Comments:{candidate_list}\n"
    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]


def comment_gen_system_prompt():
    
    sys_prompt = f"Assume you have a deep understanding of Java programming language concepts such as loops, conditionals, functions, and data structures.\nGiven a Java function '###Code', generate a short comment in one sentence.\nPlace your generated comment between <Repair> </Repair> tags.\nLet's think step by step!"

    return sys_prompt

def comment_gen_user_prompt(code):
    
    user_prompt = f"###Code:\n{code.strip()}\n"

    return user_prompt

def comment_gen_demo_prompt(data: Comment_models.Comment_Datapoint,
                                with_commands: bool,
                                is_str: bool = False) -> str:
    
    def get_demon(data: Comment_models.Comment_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:

            query = comment_gen_user_prompt(data.code)

            response = f"""<Repair>{data.nl.strip()}</Repair>"""

            return [(query, response)]
        
    if with_commands and not is_str:
        return get_demon(data)
    
    elif with_commands and is_str:
        final_result = ""
        demo_content = get_demon(data)

        for query, response in demo_content:
            final_result += query + "\n" + response

        return final_result


def comment_repair_system_prompt():

    sys_prompt = f"Given a Java function '###Code', analyze the function to understand its functionality and then select the most appropriate and concise comment from the given comment list '###Comments'.\nPut the chosen comment between <Repair> </Repair> tags.\nLet's think step by step!"

    return sys_prompt


def comment_repair_user_prompt(code, candidates: List[str], mode="random"):

    if mode == "random":
        candidate_list = list(set(candidates))
        # 打乱顺序
        random.shuffle(candidate_list)

    else:
        #去重，但是要保留原始的顺序
        candidate_list = []
        [candidate_list.append(x) for x in candidates if x not in candidate_list]

    user_prompt = f"###Code:\n{code.strip()}\n###Comments:{candidate_list}\n"

    return user_prompt


def comment_repair_demo_prompt(data: Comment_models.Comment_Datapoint,
                               with_commands: bool,
                               is_str: bool = False) -> str:
    '''
    这里的data是一个train datapoint, prediction为beam size生成的assertion
    data.nl为ground truth
    '''

    def get_demon(data: Comment_models.Comment_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:
            prediction = data.prediction

            prediction.append(data.nl.strip())

            query = comment_repair_user_prompt(data.code,
                                               prediction,
                                               mode="random")

            response = f"""<Repair>{data.nl.strip()}</Repair>"""

            return [(query, response)]

    if with_commands and not is_str:
        return get_demon(data)

    elif with_commands and is_str:
        final_result = ""
        demo_content = get_demon(data)

        for query, response in demo_content:
            final_result += query + "\n" + response

        return final_result




class OpenAIClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClientSingleton, cls).__new__(cls)
            # Initialize the OpenAI client here
            api_key= ""
            cls._instance.client = OpenAI(
                base_url="",
                api_key=api_key,
                http_client=httpx.Client(
                    base_url="",
                    follow_redirects=True,
            ),
            )
        return cls._instance

    @classmethod
    def get_client(cls):
        return cls()._instance.client

    @classmethod
    def get_response(cls,
                     messages,
                     temperature=0.7,
                     model="gpt-3.5-turbo",
                     n=1,
                     logprobs=True):
        if cls._instance is None:
            client = cls().get_client()
        else:
            client = cls._instance.client
        response = client.chat.completions.create(model=model,
                                                  messages=messages,
                                                  temperature=temperature,
                                                  n=n,
                                                  logprobs=logprobs)
        return response

class vLLMClientSingleton:
    client = None
    tokenizer = None

    def __new__(cls, model_name, *args, **kwargs):
        if cls.client is None:
            print(f"Creating a new instance for {model_name}")
            cls._instance = super(vLLMClientSingleton, cls).__new__(cls)
            cls.client = LLM(model=model_name, trust_remote_code=True, max_model_len=16384, disable_logprobs_during_spec_decoding=False)
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls.client
    
    @classmethod
    def get_response(self,
                     messages,
                     temperature=0.7,
                     model="gpt-3.5-turbo",
                     n=1,
                     logprobs=True):
        sampling_params = SamplingParams(temperature=temperature, top_p=1, n=n, logprobs=logprobs, max_tokens=1024)
        prompts = self.tokenizer.apply_chat_template(messages, tokenize=False)
        response = self.client.generate(prompts, sampling_params, use_tqdm=False)
        return response