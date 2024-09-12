import random
import lizard
from transformers import GPT2TokenizerFast
import tiktoken as tt

# tokenizer = tt.encoding_for_model("gpt-3.5-turbo")

encoding_name = 'cl100k_base'
tokenizer = tt.get_encoding(encoding_name)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_random_false_stmt():
    res = [random.choice(["true", "false"]) for x in range(10)]
    res.append("false")
    res_str = " && ".join(res)
    return res_str

def get_branch_if_else_mutant():
    mutant = 'if ('+get_random_false_stmt()+') {' + \
        get_random_type_name_and_value_statment() + \
        '}' + \
        'else{' + \
        get_random_type_name_and_value_statment() + \
        '}'
    return mutant

def get_radom_var_name():
    res_string = ''
    for x in range(8):
        res_string += random.choice('abcdefghijklmnopqrstuvwxyz')
    return res_string

def get_random_int(min, max):
    return random.randint(min, max)

def get_random_type_name_and_value_statment():
    datatype = random.choice(
        'byte,short,int,long,float,double,boolean,char,String'.split(','))
    var_name = get_radom_var_name()

    if datatype == "byte":
        var_value = get_random_int(-128, 127)
    elif datatype == "short":
        var_value = get_random_int(-10000, 10000)
    elif datatype == "boolean":
        var_value = random.choice(["true", "false"])
    elif datatype == "char":
        var_value = str(random.choice(
            'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')))
        var_value = '"'+var_value+'"'
    elif datatype == "String":
        var_value = str(get_radom_var_name())
        var_value = '"'+var_value+'"'
    else:
        var_value = get_random_int(-1000000000, 1000000000)

    mutant = str(datatype) + ' ' + str(var_name) + ' = ' + str(var_value)+";"
    return mutant

def get_branch_while_mutant():
    mutant = 'while ('+get_random_false_stmt()+') {' + \
        get_random_type_name_and_value_statment() + \
        '}'
    return mutant

def get_branch_if_mutant():
    mutant = 'if ('+get_random_false_stmt()+') {' + \
        get_random_type_name_and_value_statment() + \
        '}'
    return mutant

def get_dead_for_condition():
    var = get_radom_var_name()
    return "int "+var+" = 0; "+var+" < 0; "+var+"++"

def get_branch_for_mutant():
    dead_for_condition = get_dead_for_condition()
    mutant = 'for  ('+dead_for_condition+') {' + \
        get_random_type_name_and_value_statment() + \
        '}'
    return mutant


def get_branch_switch_mutant():
    var_name = get_radom_var_name()
    mutant = 'int ' + var_name+' = 0;' +\
        'switch  ('+var_name+') {' + \
        'case 1:' + \
        get_random_type_name_and_value_statment() + \
        'break;' +\
        'case 2:' + \
        get_random_type_name_and_value_statment() + \
        'break;' +\
        'default:' + \
        get_random_type_name_and_value_statment() + \
        'break;' +\
        '}'
    return mutant

def dead_branch_if_else(data):
    start_index = data.index("{") + 1
    start_statements = data[:start_index]
    end_statements = data[start_index:]

    mutant = get_branch_if_else_mutant()

    mutate_data = start_statements + mutant + end_statements

    return mutate_data

def dead_branch_while(data):
    start_index = data.index("{") + 1
    start_statements = data[:start_index]
    end_statements = data[start_index:]

    mutant = get_branch_while_mutant()

    mutate_data = start_statements + mutant + end_statements

    return mutate_data


def dead_branch_if(data):
    start_index = data.index("{") + 1
    start_statements = data[:start_index]
    end_statements = data[start_index:]

    mutant = get_branch_if_mutant()

    mutate_data = start_statements + mutant + end_statements

    return mutate_data


def dead_branch_for(data):
    start_index = data.index("{") + 1
    start_statements = data[:start_index]
    end_statements = data[start_index:]

    mutant = get_branch_for_mutant()

    mutate_data = start_statements + mutant + end_statements

    return mutate_data


def dead_branch_switch(data):
    start_index = data.index("{") + 1
    start_statements = data[:start_index]
    end_statements = data[start_index:]

    mutant = get_branch_switch_mutant()

    mutate_data = start_statements + mutant + end_statements

    return mutate_data



def count_codex_tokens(input: str):
    """
    word count does not equate to token count for gpt models.
    So we need to get token count for a string using the gpt tokenizers.
    https://beta.openai.com/tokenizer
    """
    res = tokenizer.encode(input.strip())
    return len(res)


def completions_create_tokens(prompt: str, max_tokens: int, n: int = 1) -> int:
    """
    Returns the upper bound of total tokens consumed for a completions.create() request.

    The formula is

      prompt_tokens + n * max_tokens

    Note that completions can be shorter than max_tokens if a sample produces a stop sequence. In this case the total tokens are lower than the above estimate.
    """
    prompt_tokens = tokenizer.encode(prompt)
    return len(prompt_tokens) + n * max_tokens


def engines_generate_tokens(context: str, length: int, completions: int = 1) -> int:
    """
    Code taken from https://replit.com/@NikolasTezak/API-TokenLoadEstimator#main.py
    Returns the upper bound of total tokens consumed for a engines.generate() request.

    This is an older but equivalent API call to completions.create(), see above.

    here max_tokens is in the in predictions response
    """
    return completions_create_tokens(
        prompt=context,
        max_tokens=length,
        n=completions,
    )


def word_count(query):
    """count the number of words in a string"""
    return len(query.split())

def complexity_count(code: str) -> str:
    i = lizard.analyze_file.analyze_source_code("./test.java", code)

    complexity = i.average_cyclomatic_complexity
    complexity_type = ''
    if complexity >= 1 and complexity <= 10:
        complexity_type = 'simple'
    elif complexity >= 11 and complexity <= 20:
        complexity_type = 'complex'
    elif complexity >= 21 and complexity <= 50:
        complexity_type = 'too complex'

    return complexity_type

def getJaccard(list1, list2):
    list1 = list(set(list1))
    list2 = list(set(list2))
    intersectionList = []
    for l in list1:
        if l in list2:
            intersectionList.append(l)
    tempList = []
    tempList.extend(list1)
    tempList.extend(list2)
    unionList = list(set(tempList))
    return len(intersectionList) * 1.0 / len(unionList)


if __name__ == "__main__":
    print(count_codex_tokens("Hello world vancouver"))
    print(engines_generate_tokens("Hello world vancouver", length=1000, completions=1))
