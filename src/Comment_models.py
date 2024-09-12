from enum import Enum
from collections import namedtuple

comment_mode = Enum('comment_mode',
                  'zero_shot '
                  'bm_25 '
                  'jarcard '
                  'zero_shot_repair')

class comment_datapoint_with_demo_length:
    def __init__(self, datapoint, token_count: int):
        self.datapoint = datapoint
        self.token_count = token_count


class Comment_Datapoint:

    def __init__(self,
                 code,
                 nl,
                 complexity,
                 prediction=None,
                 candidate=None):
        '''
        Train datapoint: generation 是多个候选的集合
        Test datapoint: candidate 是一个候选， generation是贪心搜索的结果
        '''
        self.code = code
        self.nl = nl
    
        self.complexity = complexity
        if prediction is not None:
            self.prediction = prediction
        else:
            self.prediction = []
        if candidate is not None:
            self.candidate = candidate
        else:
            self.candidate = []

    def __str__(self) -> str:
        return f"code: {self.code}\n" \
               f"nl: {self.nl}\n" \
               f"complexity: {self.complexity}\n" \
               f"prediction: {self.prediction}\n" \
               f"candidate: {self.candidate}\n"
