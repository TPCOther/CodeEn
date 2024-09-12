import os
import random
import re
import httpx
from openai import OpenAI
from antlr4 import *
from typing import List

from utils.utils import getJaccard
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import NBF_models


def nbf_get_zero_shot_prompt(bug, rule_id):

    sys_prompt = "Fix ESLint error in the following JavaScript code. Place your generated solution between <Repair> and </Repair> tags."

    temp = bug.split("[warning_line]")
    warning_line = temp[1].strip()
    temp = temp[0].split("[evidence]")
    evidence = temp[1].strip()
    bug = temp[0].split("[source_code]")[1].strip()

    user_prompt = f"###Buggy_JavaScript:\n{bug.strip()}\n###rule_id:{rule_id.strip()}\n###evidence:{evidence.strip()}\n###warning_line:{warning_line.strip()}\n"

    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]


def nbf_get_zero_shot_prompt_repair(bug, rule_id,
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

    sys_prompt = "Identyfy the correct fix from a set of proposed solutions labeled ###Fix_Code to fix ESLint error in the following JavaScript code. Place your chosen solution between <Repair> and </Repair> tags."

    temp = bug.split("[warning_line]")
    warning_line = temp[1].strip()
    temp = temp[0].split("[evidence]")
    evidence = temp[1].strip()
    bug = temp[0].split("[source_code]")[1].strip()

    user_prompt = f"###Buggy_JavaScript:\n{bug.strip()}\n###rule_id:{rule_id.strip()}\n###evidence:{evidence.strip()}\n###warning_line:{warning_line.strip()}\n###Fix_Code:{candidate_list}\n"
    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]



def nbf_repair_system_prompt():


    sys_prompt = "Identyfy the correct fix from a set of proposed solutions labeled ###Fix_Code to fix ESLint error in the following JavaScript code. Place your chosen solution between <Repair> and </Repair> tags."

    return sys_prompt


def nbf_repair_user_prompt(bug, rule_id, candidates: List[str], mode="random"):

    if mode == "random":
        candidate_list = list(set(candidates))
        # 打乱顺序
        random.shuffle(candidate_list)

    else:
        #去重，但是要保留原始的顺序
        candidate_list = []
        [candidate_list.append(x) for x in candidates if x not in candidate_list]

    temp = bug.split("[warning_line]")
    warning_line = temp[1].strip()
    temp = temp[0].split("[evidence]")
    evidence = temp[1].strip()
    bug = temp[0].split("[source_code]")[1].strip()

    user_prompt = f"###Buggy_JavaScript:\n{bug.strip()}\n###rule_id:{rule_id.strip()}\n###evidence:{evidence.strip()}\n###warning_line:{warning_line.strip()}\n###Fix_Code:{candidate_list}\n"

    return user_prompt

def nbf_gen_system_prompt():

    sys_prompt = "Fix ESLint error in the following JavaScript code. Place your generated solution between <Repair> and </Repair> tags."

    return sys_prompt

def nbf_gen_user_prompt(bug, rule_id):

    temp = bug.split("[warning_line]")
    warning_line = temp[1].strip()
    temp = temp[0].split("[evidence]")
    evidence = temp[1].strip()
    bug = temp[0].split("[source_code]")[1].strip()

    user_prompt = f"###Buggy_JavaScript:\n{bug.strip()}\n###rule_id:{rule_id.strip()}\n###evidence:{evidence.strip()}\n###warning_line:{warning_line.strip()}\n"

    return user_prompt

def nbf_gen_demo_prompt(data: NBF_models.NBF_Datapoint,
                       with_commands: bool,
                          is_str: bool = False) -> str:
    def get_demon(data: NBF_models.NBF_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:

            query = nbf_gen_user_prompt(data.bug, data.rule_id)

            response = f"""<Repair>{data.fix.strip()}</Repair>"""

            return [(query, response)]
    
    if with_commands and not is_str:
        return get_demon(data)
    
    elif with_commands and is_str:
        final_result = ""
        demo_content = get_demon(data)

        for query, response in demo_content:
            final_result += query + "\n" + response

        return final_result



def nbf_repair_demo_prompt(data: NBF_models.NBF_Datapoint,
                               with_commands: bool,
                               is_str: bool = False) -> str:
    '''
    这里的data是一个train datapoint, prediction为beam size生成的assertion
    data.nl为ground truth
    '''

    def get_demon(data: NBF_models.NBF_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:
            prediction = data.prediction

            prediction.append(data.fix.strip())

            query = nbf_repair_user_prompt(data.bug,
                                               data.rule_id,
                                               prediction,
                                               mode="random")

            response = f"""<Repair>{data.fix.strip()}</Repair>"""

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
            # api_key="sk-jGs9mtkkfOB84yxhD787C94c70B04f6cB28dE4415d462b80"
            # cls._instance.client = OpenAI(
            #     base_url="https://www.jcapikey.com/v1",
            #     api_key=api_key,
            #     http_client=httpx.Client(
            #         base_url="https://www.jcapikey.com/v1",
            #         follow_redirects=True,
            #     ),
            # )
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
        #临时加入的model参数
        model = "gpt-3.5-turbo"
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
        response = self.client.generate(prompts, sampling_params)
        return response


        
