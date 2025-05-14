import abc
import numpy as np
import torch
from tqdm import tqdm
from benchmark.NBF.ms_evaluate import INS, generate_prompt, read_examples_, unix_convert_examples_to_features
from transformers import GenerationConfig

def outer_product_opt(
        c1,
        d1,
        c2,
        d2
):
    """Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products."""
    b1, b2 = c1.shape[0], c2.shape[0]
    t1 = np.matmul(
        np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)),
        np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1))
    )
    t2 = np.matmul(
        np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)),
        np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1))
    )
    t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
    t1 = t1.reshape(b1, 1).repeat(b2, axis=1)
    t2 = t2.reshape(1, b2).repeat(b1, axis=0)
    return t1 + t2 - 2 * t3

def kmeans_plus_plus_opt(
        x1_list,
        x2_list,
        n_clusters,
        init,
        random_state = np.random.RandomState(1234),
        n_local_trials = None
):
    """Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)."""

    idxs = np.empty((n_clusters+len(init)-1,), dtype=np.int64)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))
    # Pick first center randomly
    idxs[:len(init)] = init
    # Initialize list of closest distances and calculate current potential
    distance_to_candidates_list = []
    for x1, x2 in zip(x1_list, x2_list):
        distance_to_candidates_list.append(outer_product_opt(
            x1[init], x2[init], x1, x2
        ).reshape(len(init), -1))
    distance_to_candidates = sum(distance_to_candidates_list)
    candidates_pot = distance_to_candidates.sum(axis=1)
    best_candidate = np.argmin(candidates_pot)
    current_pot = candidates_pot[best_candidate]
    closest_dist_sq = distance_to_candidates[best_candidate]
    # Pick the remaining n_clusters-1 points
    for c in range(len(init), len(init)+n_clusters-1):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
        # Compute distances to center candidates
        distance_to_candidates_list = []
        for x1, x2 in zip(x1_list, x2_list):
            distance_to_candidates_list.append(outer_product_opt(
                x1[candidate_ids], x2[candidate_ids], x1, x2
            ).reshape(len(candidate_ids), -1))
        distance_to_candidates = sum(distance_to_candidates_list)
        # update closest distances squared and potential for each candidate
        np.minimum(
            closest_dist_sq, distance_to_candidates, out=distance_to_candidates
        )
        candidates_pot = distance_to_candidates.sum(axis=1)
        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]
        idxs[c] = best_candidate
    return idxs[len(init)-1:]

def get_real_length(sequences, stop):
    max_len = sequences[0].shape[0]
    real_length = []
    for seq in sequences:
        non_zero_indices = torch.nonzero(seq == stop)
        first_zero_index = non_zero_indices[0] if len(non_zero_indices) > 0 else max_len
        real_length.append(first_zero_index)
    return torch.tensor(real_length).cpu()

def calculate_autoregressive_ppl(model_outputs, real_lengths, generated_sequences):
    """
    使用自回归方式计算PPL（将生成的序列向左移动一位作为目标）
    
    参数:
    - model_outputs: 模型输出的概率分布列表
    - real_lengths: 每个序列的实际长度列表
    - generated_sequences: 模型生成的序列
    
    返回:
    - PPL列表
    """
    all_ppl_scores = []
    
    for batch_idx, batch_output in enumerate(model_outputs):
        batch_lengths = real_lengths[batch_idx]
        
        # 用于调试的打印，可以保留
        print(f"Batch {batch_idx} - Output shape: {batch_output.shape}")
        print(f"Batch {batch_idx} - Lengths shape: {real_lengths[batch_idx].shape}")
        print(f"Batch {batch_idx} - Sequences shape: {generated_sequences[batch_idx].shape}")
        
        for i in range(len(batch_output)):
            # 确保长度在有效范围内
            length = batch_lengths[i].item() if hasattr(batch_lengths[i], 'item') else batch_lengths[i]
            
            # 确保我们有至少两个token (一个作为输入，一个作为目标)
            if length <= 1:
                all_ppl_scores.append(float('inf'))
                continue
            
            # 获取当前序列的预测概率
            seq_probs = batch_output[i][:length-1]  # 除去最后一个token
            
            # 获取当前序列
            seq_idx = i  # 如果批次和序列是一一对应的
            # 如果generated_sequences是按批次存储的，则使用正确的索引
            if batch_idx < len(generated_sequences):
                target_seq = generated_sequences[batch_idx][seq_idx] if generated_sequences[batch_idx].ndim > 1 else generated_sequences[batch_idx]
                target_ids = target_seq[1:length]  # 从第二个token开始
            else:
                # 处理不同批次结构的情况
                target_seq = generated_sequences[i]
                target_ids = target_seq[1:length]
            
            # 获取目标token的概率
            token_probs = []
            for j, target_id in enumerate(target_ids):
                if j < seq_probs.size(0):  # 确保j在seq_probs的范围内
                    token_id = target_id.item() if hasattr(target_id, 'item') else target_id
                    if token_id < seq_probs.size(1):  # 确保token_id在词表大小范围内
                        prob = seq_probs[j][token_id].item()
                        token_probs.append(prob)
            
            # 计算困惑度
            if token_probs:
                log_probs = [np.log(p) if p > 0 else -float('inf') for p in token_probs]
                valid_log_probs = [lp for lp in log_probs if lp != -float('inf')]
                if valid_log_probs:
                    mean_log_prob = np.mean(valid_log_probs)
                    ppl = np.exp(-mean_log_prob)
                else:
                    ppl = float('inf')
            else:
                ppl = float('inf')
                
            all_ppl_scores.append(ppl)
    
    return all_ppl_scores

def get_model_output(test_set, test_objects, model, tokenizer, device, model_under_test, task=None, return_sequences=False):
    '''
    获取模型输出
    '''
    print(model_under_test)
    if model_under_test.lower() == "codegen":
        batch_size = 8
    elif model_under_test.lower() == "unixcoder":
        batch_size = 16
    else:
        batch_size = 128
        
    print(batch_size)
    if task == "NBF":
        datasets = test_set.bug
        from benchmark.NBF.ms_evaluate import INS, generate_prompt, read_examples_, unix_convert_examples_to_features
    else:
        datasets = test_set.source_dataset
    
    if task == "ATLAS":
        from benchmark.ATLAS.ms_evaluate import INS, generate_prompt, read_examples_, unix_convert_examples_to_features
    if task == "JCSD":
        from benchmark.JCSD.ms_evaluate import INS, generate_prompt, read_examples_, unix_convert_examples_to_features
    objects = test_objects

    model_output = []
    real_lengths = []
    sequences = []
    
    for i in tqdm(range(0, len(datasets), batch_size)):
        model_input = datasets[i:i + batch_size]
        model_objects = objects[i:i + batch_size]
        softmax = torch.nn.functional.softmax

        if model_under_test == "CodeT5":
            inputs = tokenizer(model_input,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt").to(device)
        
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids,
                        max_length=128,
                        do_sample=False,  # 关闭采样，得到最高概率的输出
                        return_dict_in_generate=True,
                        output_scores=True)
                seq_len = get_real_length(outputs.sequences.squeeze(1), tokenizer.eos_token_id)
                probs = [softmax(scores, dim=-1) for scores in outputs.scores]
                probs_stacked = torch.stack(probs)
                probs_transposed = probs_stacked.transpose(0, 1).cpu()

            model_output.append(probs_transposed)
            real_lengths.append(seq_len)
            sequences.append(outputs.sequences.cpu())
            
            
        
        elif model_under_test == "CodeGen":
            input_data = [
                generate_prompt(INS, input)
                for input in model_input
            ]
            inputs = tokenizer(input_data,
                                max_length=1920,
                                truncation=True,
                                padding=True,
                                return_tensors="pt").to(device)  
            
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = model.generate(
                    inputs.input_ids, max_new_tokens=128, do_sample=False, return_dict_in_generate=True, output_scores=True, generation_config=generation_config)
            source_length = inputs.input_ids.shape[1]
            seq_len = get_real_length(outputs.sequences[:, source_length:], tokenizer.pad_token_id)
            probs = [softmax(scores, dim=-1) for scores in outputs.scores]
            probs_stacked = torch.stack(probs)
            probs_transposed = probs_stacked.transpose(0, 1).cpu()

            model_output.append(probs_transposed)
            real_lengths.append(seq_len)
            sequences.append(outputs.sequences.cpu())

        if model_under_test == "Unixcoder":
            
            examples = read_examples_(model_objects)

            features = unix_convert_examples_to_features(
                examples,
                tokenizer,
                max_source_length=512,
                max_target_length=128)

            source_ids = torch.tensor([f.source_ids for f in features],
                                    dtype=torch.long).to(device)
            attention_mask = source_ids.ne(tokenizer.pad_token_id)

            with torch.no_grad():
                outputs = model(source_ids, return_logits=True)
                seq_len = get_real_length(outputs.sequences.squeeze(1), tokenizer.bos_token_id)
                probs = [softmax(scores, dim=-1) for scores in outputs.scores]
                probs_transposed = torch.stack(probs).cpu()

                model_output.append(probs_transposed)
                real_lengths.append(seq_len)
                sequences.append(outputs.sequences.cpu())

    if return_sequences:
        return model_output, real_lengths, sequences
    else:
        return model_output, real_lengths

class SamplingMethod(object):
    """Base class for sampling methods."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, n, debug_info, model_type, device):
        self.n = n
        self.all_indices = np.arange(self.n, dtype=np.int64)
        self.scores = None
        self.debug_info = debug_info
        self.model_type = model_type
        self.device = device

    @abc.abstractmethod
    def get_scores(
            self,
            already_selected_indices,
            label_budget
    ):
        """Gets scores of the test data for sampling."""
        return np.zeros(self.n, dtype=np.float32)

    def select_batch_to_label(
            self,
            already_selected_indices,
            label_budget,
            update_scores=True,
            task=None
    ):
        """Returns the indices of batch of samples to label.

        Args:
          already_selected_indices: index of datapoints already selected
          label_budget: labeling budget
          update_scores: whether to update the scores

        Returns:
          indices of samples selected to label
        """
        if (self.scores is None) or update_scores:
            self.scores = self.get_scores(already_selected_indices, label_budget, task=task)
        remain_indices = np.setdiff1d(self.all_indices, already_selected_indices)
        sorted_index = np.argsort(self.scores[remain_indices])
        remain_indices = remain_indices[sorted_index]
        newly_selected_indices = remain_indices[:label_budget]
        selected_indices = np.concatenate(
            (already_selected_indices, newly_selected_indices), axis=0
        )
        if self.debug_info:
            min_score = np.min(self.scores[newly_selected_indices])
            max_score = np.max(self.scores[newly_selected_indices])
            print(
                f'Min selected scores: {min_score}, max selected score: {max_score}'
            )
        return selected_indices



class BADGESampling(SamplingMethod):
    """BADGE sampling method."""

    def __init__(
            self,
            features,
            tset,
            tobject,
            ensemble_models,
            n,
            tokenizer,
            model_type,
            device,
            ensemble_method='soft',
            random_seed=1234,
            debug_info=False
    ):
        super().__init__(n=n, debug_info=debug_info, model_type=model_type, device=device)
        self.ensemble_models = ensemble_models
        self.ensemble_method = ensemble_method
        self.random_seed = random_seed
        self.tokenizer = tokenizer
        self.features = features
        self.tset = tset
        self.tobject = tobject

    def get_scores(
            self,
            already_selected_indices,
            label_budget,
            task = None
    ):
        """Gets scores of the test data for sampling."""
        if already_selected_indices.shape[0] + label_budget >= self.n:
            # Scores are useless in this case,
            # since all remaining samples will be selected.
            return np.zeros(self.n, dtype=np.float32)
        remain_indices = np.setdiff1d(self.all_indices, already_selected_indices)
        remain_size = remain_indices.shape[0]
        num_classes = len(self.tokenizer)
        remain_feature_list = []
        uncertain_score_list = []
        for model in self.ensemble_models:
            uncertain_scores = []
            outputs, seq_lens = get_model_output(self.tset, self.tobject, model, self.tokenizer, self.device, self.model_type, task=task)

            for batch_output, seq_len in zip(outputs, seq_lens):
                for i in range(len(batch_output)):
                    np_outputs = batch_output[i][:seq_len[i]].cpu().numpy()
                    preds = np.argmax(np_outputs, axis=1)
                    length = min(seq_len[i], preds.shape[0])
                    scores_delta = np.zeros((length, num_classes), dtype=np.float32)
                    scores_delta[np.arange(length), preds] = 1.0
                    score = np.sum(np_outputs - scores_delta, axis=0)
                    uncertain_scores.append(score / scores_delta.shape[0])

            features = np.array([f.tolist() for f in self.features])
            remain_features = features[remain_indices]
            remain_uncertain_scores = np.stack([uncertain_scores[i] for i in remain_indices], axis=0)
            remain_feature_list.append(remain_features)
            uncertain_score_list.append(remain_uncertain_scores)
        random_state = np.random.RandomState(self.random_seed)
        init = np.array([random_state.randint(remain_size)])
        q_idxs = kmeans_plus_plus_opt(
            x1_list=uncertain_score_list,
            x2_list=remain_feature_list,
            n_clusters=label_budget,
            init=init,
            random_state=random_state,
            n_local_trials=None,
        )
        scores = np.ones(self.n, dtype=np.float32)
        scores[remain_indices[q_idxs]] = 0
        return scores