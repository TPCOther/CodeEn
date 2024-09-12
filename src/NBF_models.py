from enum import Enum
from collections import namedtuple

nbf_mode = Enum('bf_mode',
                  'zero_shot '
                  'bm_25 '
                  'jarcard '
                  'zero_shot_repair')

class nbf_datapoint_with_demo_length:
    def __init__(self, datapoint, token_count: int):
        self.datapoint = datapoint
        self.token_count = token_count


class NBF_Datapoint:

    def __init__(self,
                 bug,
                 fix,
                 rule_id,
                 warning_line,
                 prediction=None,
                 candidate=None):
        '''
        Train datapoint: generation 是多个候选的集合
        Test datapoint: candidate 是一个候选， generation是贪心搜索的结果
        '''
        self.bug = bug
        self.fix = fix
        self.warning_line = warning_line
        self.rule_id = rule_id

        if prediction is not None:
            self.prediction = prediction
        else:
            self.prediction = []
        if candidate is not None:
            self.candidate = candidate
        else:
            self.candidate = []

    def __str__(self) -> str:
        return f"bug: {self.bug}\nfix: {self.fix}\nprediction: {self.prediction}\ncandidate: {self.candidate}\n"
