# encoding=utf-8

"""
nltk: nature language Toolkit
     1) NLTK 分析单词和句子:
        from nltk.tokenize import sent_tokenize, word_tokenize
     2) NLTK 与停止词:
        from nltk.corpus import stopwords
        set(stopwords.words('english'))
     3) NLTK 词干提取: 词干的概念是一种规范化方法。 除涉及时态之外，许多词语的变体都具有相同的含义。
            from nltk.stem import PorterStemmer
            from nltk.tokenize import sent_tokenize, word_tokenize
            ps = PorterStemmer()
            ps.stem(w)
     4)


"""

import re
import nltk
from typing import List

from utils.javatokenizer.tokenizer import tokenize_java_code_o
from difflib import get_close_matches

'''
 初步分析为NLP的Token；Javatokenizer为Code Token
'''


class Tokenizer:

    @classmethod
    def camel_case_split(cls, identifier):
        return re.sub(r'([A-Z][a-z])', r' \1',
                      re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()

    @classmethod
    def tokenize_identifier_raw(cls, token, keep_underscore=True):
        '''
        标识符存在_下划线，对其进行处理，下划线保留，驼峰再切割
        '''
        regex = r'(_+)' if keep_underscore else r'_+'  # split函数：如果有(), 则同时返回()，若没有 咋不返回
        id_tokens = []
        for t in re.split(regex, token):
            if t:
                id_tokens += cls.camel_case_split(t)
        return list(filter(lambda x: len(x) > 0, id_tokens))

    @classmethod
    def tokenize_desc_with_con(cls, desc: str) -> List[str]:
        '''
        自然语言的分割： 1) 空格分割 2) nltk.word_tokenize再分割 3) 驼峰法则再分割
        '''

        def _tokenize_word(word):
            new_word = re.sub(r'([-!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])',
                              r' \1 ', word)
            subwords = nltk.word_tokenize(new_word)
            new_subwords = []
            for w in subwords:
                new_subwords += cls.tokenize_identifier_raw(
                    w, keep_underscore=True)
            return new_subwords

        tokens = []
        for word in desc.split():
            if not word:
                continue
            tokens += " <con> ".join(_tokenize_word(word)).split()
        return tokens

    @classmethod
    def whether_equally(cls, expected, actual):
        '''
        判断两个token是否相等
        '''

        def _strip_full_form(assertion):
            listical = list(assertion.split())
            final_str = ''
            found_assertion = False
            for i in range(len(listical)):
                if listical[i].startswith('assert'):
                    found_assertion = True
                if found_assertion:
                    final_str += listical[i] + ' '
            return final_str

        def _strip_extra_parenthesis(expected):
            if '( (' in expected and ') )' in expected:
                expected = expected.replace('( (', '(')
                expected = expected.replace(') )', ')')

        def _replace_assert_true_false_assert_equal(expected, actual):
            ASSERT_EQUALS_TRUE = 'assertEquals ( true ,'
            ASSERT_EQUALS_FALSE = 'assertEquals ( false ,'
            ASSERT_TRUE = 'assertTrue ('
            ASSERT_FALSE = 'assertFalse ('
            if (ASSERT_EQUALS_TRUE in expected and ASSERT_TRUE in actual) or \
                    ASSERT_EQUALS_TRUE in actual and ASSERT_TRUE in expected:
                expected = expected.replace(ASSERT_EQUALS_TRUE, ASSERT_TRUE)
                actual = actual.replace(ASSERT_EQUALS_TRUE, ASSERT_TRUE)
            elif (ASSERT_EQUALS_FALSE in expected and ASSERT_FALSE in actual) or \
                    ASSERT_EQUALS_FALSE in actual and ASSERT_FALSE in expected:
                expected = expected.replace(ASSERT_EQUALS_FALSE, ASSERT_FALSE)
                actual = actual.replace(ASSERT_EQUALS_FALSE, ASSERT_FALSE)

        def _match_args(expected, actual):

            def find_match(text):
                x = re.findall("\(\s*([^)]+?)\s*\)", text)
                if len(x):
                    return [a.strip() for single in x for a in single.split(',')]
                return []

            def get_assertion_type(text):
                for c in text.split():
                    if c.startswith('assert'):
                        return c

            expected_args = sorted(find_match(expected))
            actual_args = sorted(find_match(actual))

            expected_assertion_type = get_assertion_type(expected)
            actual_assertion_type = get_assertion_type(actual)
            return len(expected_args) and len(actual_args) and \
                expected_args == actual_args and expected_assertion_type == actual_assertion_type

        tokens = tokenize_java_code_o(expected)
        tokens = [t.text for t in tokens]

        out_tokens = tokenize_java_code_o(actual)
        out_tokens = [t.text for t in out_tokens]

        if tokens == out_tokens:
            return True

        expected = _strip_full_form(expected)
        actual = _strip_full_form(actual)

        _strip_extra_parenthesis(expected=expected)

        tokens = tokenize_java_code_o(expected)
        tokens = [t.text for t in tokens]

        out_tokens = tokenize_java_code_o(actual)
        out_tokens = [t.text for t in out_tokens]

        if tokens == out_tokens:
            return True

        _replace_assert_true_false_assert_equal(expected=expected,
                                                actual=actual)

        tokens = tokenize_java_code_o(expected)
        tokens = [t.text for t in tokens]

        out_tokens = tokenize_java_code_o(actual)
        out_tokens = [t.text for t in out_tokens]

        if tokens == out_tokens:
            return True

        if _match_args(expected=expected, actual=actual):
            return True

        return False

    @classmethod
    def NBFS_match(cls, expected, actual, buggy_code, warning_line):
        
        def is_exact_match(expected, actual):
            expected = ''
            for line in expected.split('\n'):
                expected += re.sub(' +', ' ', line.strip()).replace(' ', '')
            
            codex = ''
            for line in actual.split('\n'):
                codex += re.sub(' +', ' ', line.strip()).replace(' ', '')

            return expected == codex or expected in codex

        def is_match(expected_orig, actual_orig, buggy_code, warning_line):
            if is_exact_match():
                return True
            
            splitted_buggy_code = buggy_code.split('\n')
            splitted_expected_code = expected_orig.split('\n')
            splitted_codex_code = actual_orig.split('\n')

            expected = ''
            for line in expected_orig.split('\n'):
                expected += re.sub(' +', ' ', line.strip())
            
            actual = ''
            for line in actual_orig.split('\n'):
                actual += re.sub(' +', ' ', line.strip())

            if len(splitted_buggy_code) > len(splitted_expected_code):
                closest_match_expected = get_close_matches(warning_line, splitted_expected_code, n=1)
                if not len(closest_match_expected): ## The warning line was completely removed
                    if warning_line not in expected and warning_line not in actual:
                        return True
                else: ## The warning line exists but is modified
                    closest_match_codex = get_close_matches(warning_line, splitted_codex_code, n=1)
                    if len(closest_match_codex) == len(closest_match_expected) == 1:
                        if closest_match_expected[0] \
                            .replace(' ', '') \
                            .replace('==', '===') \
                            .replace('(', '').replace(')', '') == closest_match_codex[0] \
                            .replace(' ', '') \
                            .replace('==', '===') \
                            .replace('(', '').replace(')', ''):
                            return True
            
            elif len(splitted_buggy_code) == len(splitted_expected_code) == len(splitted_codex_code):
                i = 0
                buggy_line_num = -1
                for line in splitted_buggy_code:
                    if line.strip() == warning_line:
                        buggy_line_num = i
                    i += 1
                
                if buggy_line_num != -1 and splitted_codex_code[buggy_line_num] \
                    .replace(' ', '') \
                    .replace('==', '===') \
                    .replace('(', '').replace(')', '') == splitted_expected_code[buggy_line_num] \
                    .replace(' ', '') \
                    .replace('==', '===') \
                    .replace('(', '').replace(')', ''):
                    return True 

            return False


        expected = expected.strip().lower()
        actual = actual.strip().lower()
        buggy_code = buggy_code.strip().lower()
        warning_line = warning_line.strip().lower()

        return is_exact_match(expected, actual)

    
    @classmethod
    def Recoder_whether_equally(cls, expected, actual):
        '''
        判断两个token是否相等
        '''

        tokens = tokenize_java_code_o(expected)
        tokens = [t.text for t in tokens]

        out_tokens = tokenize_java_code_o(actual)
        out_tokens = [t.text for t in out_tokens]

        return tokens == out_tokens

    @classmethod
    def Comment_whether_equally(cls, expected, actual):
        '''
        判断两个token是否相等
        '''

        tokens = tokenize_java_code_o(expected)
        tokens = [t.text for t in tokens]

        out_tokens = tokenize_java_code_o(actual)
        out_tokens = [t.text for t in out_tokens]

        return tokens == out_tokens
    @classmethod
    def Tokenize_code(cls, code: str) -> List[str]:
        '''
        对代码进行分割
        '''
        tokens = tokenize_java_code_o(code)
        return [t.text for t in tokens]

if __name__ =="__main__":
    # mytext = "Bonjour M. Adam, comment allez-vous? J'espère que tout va bien. Aujourd'hui est un bon jour."
    # print(Tokenizer.tokenize_desc_with_con(mytext))
    # print(nltk.tokenize.word_tokenize(mytext))
    pass
