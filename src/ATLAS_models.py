from enum import Enum
from collections import namedtuple

atlas_mode = Enum('atlas_mode',
                  'zero_shot '
                  'bm_25 '
                  'jarcard '
                  'zero_shot_repair')

class atlas_datapoint_with_demo_length:
    def __init__(self, datapoint, token_count: int):
        self.datapoint = datapoint
        self.token_count = token_count


class ATLAS_Datapoint:

    def __init__(self,
                 focal_method,
                 test_method,
                 focal_test,
                 assertion,
                 assertion_type,
                 method_name,
                 test_name,
                 complexity,
                 prediction=None,
                 candidate=None):
        '''
        Train datapoint: generation 是多个候选的集合
        Test datapoint: candidate 是一个候选， generation是贪心搜索的结果
        '''
        self.focal_method = focal_method
        self.test_method = test_method
        self.focal_test = focal_test
        self.assertion = assertion
        self.assertion_type = assertion_type
        self.method_name = method_name
        self.test_name = test_name
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
        return f"ATLAS_Datapoint(focal_method={self.focal_method}, test_method={self.test_method}, assertion={self.assertion}, assertion_type={self.assertion_type}, method_name={self.method_name}, test_name={self.test_name}, complexity={self.complexity}, prediction={self.prediction}, candidate={self.candidate})"
