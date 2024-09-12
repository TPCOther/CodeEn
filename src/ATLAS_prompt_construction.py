import random
import re
import httpx
from openai import OpenAI
from antlr4 import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ATLAS_models
from typing import List

from utils.javatokenizer.tokenizer import tokenize_java_code_origin, tokenize_java_code_raw
from utils.utils import getJaccard

STOP_DELIMETER = "END_OF_DEMO"

ASSERTION_TYPES = ["assertEquals", "assertTrue", "assertNotNull",
                   "assertThat", "assertNull", "assertFalse",
                   "assertArrayEquals", "assertSame"]


split_re = re.compile("\"<AssertPlaceHolder>\" ;[ }]*")


def assertion_get_zero_shot_prompt(focal_method, test_prefix):
    #编写给定focal_method,test_prefix 生成对应assertion的prompt
    sys_prompt = f"Given a focal method (###Focal_method), the corresponding test prefix (###Test_prefix), generate the corresponding assertion without any comments.\nPut the assertion between <Generation> </Generation> tags"

    user_prompt = f"###Focal_method:\n{focal_method.strip()}\n### Test_prefix:\n{test_prefix.strip()}\n"

    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]


def assertion_get_zero_shot_prompt_repair(focal_method, test_prefix,
                                          candidates: List[str], mode="random"):

    # 初始化的assertion是一个list，需要将其转换为string
    # .
    if mode == "random":
        candidate_list = list(set(candidates))
        # 打乱顺序
        random.shuffle(candidate_list)

    else:
        #去重，但是要保留原始的顺序
        candidate_list = []
        [candidate_list.append(x) for x in candidates if x not in candidate_list]

    sys_prompt = f"For the provided focal method '###Focal_method' and test prefix '###Test_prefix', identify the most appropriate assertion from the given list '###Assertions'.\nPut the chosen assertion between <Repair> </Repair> tags."

    user_prompt = f"###Focal_method:\n{focal_method.strip()}\n###Test_prefix:\n{test_prefix.strip()}\n###Assertions:{candidate_list}\n"

    return [{
        "role": "system",
        "content": sys_prompt
    }, {
        "role": "user",
        "content": user_prompt
    }]

def assertion_get_whether_repair_prompt(focal_method, test_prefix, init_assertion):


    sys_prompt = f"Given a focal method (###Focal_method), the corresponding test prefix (###Test_prefix), and an assertion candidate (###Assertion), determine if the assertion candidate is semantically and syntactically correct based on ###Focal_method and ###Test_prefix. Put the answer (CORRECT or INCORRECT) between <Ans> </Ans> tags."

    user_prompt = f"###Focal_method:\n{focal_method.strip()}\n###Test_prefix:\n{test_prefix.strip()}\n###Assertion:\n{init_assertion.strip()} ;\n"

    return [{"role": "system", "content": sys_prompt},{"role": "user", "content": user_prompt}]


def assertion_repair_system_prompt():

    sys_prompt = f"For the provided focal method '###Focal_method' and test prefix '###Test_prefix', identify the most appropriate assertion from the given list '###Assertions', based on semantic and syntactic correctness.\nPut the chosen assertion between <Repair> </Repair> tags.\nLet's think step by step!"
    
    return sys_prompt


def assertion_repair_user_prompt(focal_method, test_prefix,
                                 candidates: List[str], mode="random"):
    
    if mode == "random":
        candidate_list = list(set(candidates))
        # 打乱顺序
        random.shuffle(candidate_list)

    else:
        #去重，但是要保留原始的顺序
        candidate_list = []
        [candidate_list.append(x) for x in candidates if x not in candidate_list]
    

    user_prompt = f"###Focal_method:\n{focal_method.strip()}\n###Test_prefix:\n{test_prefix.strip()}\n###Assertions:{candidate_list}\n"

    return user_prompt


def assertion_repair_demo_prompt(data: ATLAS_models.ATLAS_Datapoint,
                                 with_commands: bool,
                                 is_str: bool = False) -> str:
    '''
    这里的data是一个train datapoint, prediction为beam size生成的assertion
    data.assertion为ground truth
    '''

    def get_demon(data: ATLAS_models.ATLAS_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:
            prediction = data.prediction

            prediction.append(data.assertion.strip())

            query = assertion_repair_user_prompt(data.focal_method,
                                                 data.test_method, prediction, mode="random")
            response = f"""<Repair>{data.assertion.strip()}</Repair>"""

            return [(query, response)]

    if with_commands and not is_str:
        return get_demon(data)

    elif with_commands and is_str:
        final_result = ""
        demo_content = get_demon(data)

        for query, response in demo_content:
            final_result += query + "\n" + response

        return final_result
    


def assertion_gen_system_prompt():

    sys_prompt = f"For the provided focal method '###Focal_method' and test prefix '###Test_prefix', generate the corresponding assertion, based on semantic and syntactic correctness.\nPut the generated assertion between <Repair> </Repair> tags.\nLet's think step by step!"
    
    return sys_prompt


def assertion_gen_user_prompt(focal_method, test_prefix):

    user_prompt = f"###Focal_method:\n{focal_method.strip()}\n###Test_prefix:\n{test_prefix.strip()}\n"

    return user_prompt


def assertion_gen_demo_prompt(data: ATLAS_models.ATLAS_Datapoint,
                                 with_commands: bool,
                                 is_str: bool = False) -> str:
    '''
    这里的data是一个train datapoint, prediction为beam size生成的assertion
    data.assertion为ground truth
    '''

    def get_demon(data: ATLAS_models.ATLAS_Datapoint):

        if len(data.prediction) == 0:
            raise Exception("No prediction")

        else:
            query = assertion_gen_user_prompt(data.focal_method,
                                                 data.test_method)
            
            response = f"""<Repair>{data.assertion.strip()}</Repair>"""

            return [(query, response)]

    if with_commands and not is_str:
        return get_demon(data)

    elif with_commands and is_str:
        final_result = ""
        demo_content = get_demon(data)

        for query, response in demo_content:
            final_result += query + "\n" + response

        return final_result




def split_focal_test_to_parts(focal_test):
    '''
    ATLAS: 将focal_test分为两部分，前半部分是test_method，后半部分是focal_method
    '''
    err_flag = False
    match = split_re.search(focal_test)
    if not match:
        err_flag = True
        return (None, None), err_flag
    idx = match.span()[1]
    test_method = focal_test[0:idx]
    focal_method = focal_test[idx:]
    return (focal_method, test_method), err_flag


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
                     model="gpt-3.5-turbo-0125",
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