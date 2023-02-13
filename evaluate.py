import json
import argparse
import math
import os
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm
import pickle as pc

from transformers import AutoModelForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric
from parlai.core.metrics import (
    FixedMetric, 
    AverageMetric, 
    IntraDistinctMetric, 
    InterDistinctMetric,
    F1Metric
)
from modules.empathy_scorer import EmpathyScorer
from nltk import word_tokenize


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

EMOTIONS = ['proud', 'apprehensive', 'disappointed', 'faithful', 'impressed', 'devastated', 'prepared', 'nostalgic', 'annoyed', 'grateful', 'joyful', 'terrified', 'caring', 'trusting', 'sad', 'guilty', 'sentimental', 'hopeful', 'confident', 'surprised', 'furious', 'afraid', 'jealous', 'excited', 'lonely', 'disgusted', 'embarrassed', 'angry', 'content', 'ashamed', 'anticipating', 'anxious']
EMPINTENTS = ['agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']


def load_word2nidf(wordcount_dir):
    """
    Loads word count stats from word2count.pkl file in the preparation dataset,
    computes NIDF for all words, and returns the word2nidf dictionary.
    
    Returns:
        word2nidf: dict mapping words to their NIDF score (float between 0 and 1)
    """
    
    word2count_fp = os.path.join(wordcount_dir, 'word2count.pkl')
    
    with open(word2count_fp, "rb") as f:
        data = pc.load(f)
    
    num_sents = data['num_sents']
    
    word2count = data['word2count']
    min_c = min(word2count.values()) # min count
    max_c = max(word2count.values()) # max count
    
    word2nidf = {
        w: (math.log(max_c) - math.log(c)) / (math.log(max_c) - math.log(min_c))
        for w, c in word2count.items()
    }
    
    return word2nidf

def get_avg_nidf(utter, word2nidf=None, word_tok=None):
    """
    Returns the mean NIDF of the words in utter.
    """
    utter = utter.lower()
    words = utter.split()

    problem_words = [w for w in words if w not in word2nidf]
    ok_words = [w for w in words if w in word2nidf]
    
    if len(ok_words) == 0:
        print(
            f"WARNING: For all the words in the utterance '{utter}', we do not have the NIDF score. Marking as avg_nidf=1."
        )
        return 1 # rarest possible sentence
    
    nidfs = [word2nidf[w] for w in ok_words]
    avg_nidf = sum(nidfs) / len(nidfs)
    avg_nidf = max(nidfs)

    if len(problem_words) > 0:
        print(
            f"WARNING: When calculating avg_nidf for the utterance '{utter}', we don't know NIDF for the following words: {str(problem_words)}"
        )
    
    assert avg_nidf >= 0 and avg_nidf <= 1
    return avg_nidf


def get_empathy_intent(data, clf_args, clf_tokenizer, clf_model, empintent_labels):
    
    label_list = list(empintent_labels.values())

    for example in tqdm(data):
        utter = example['utterance']
        pred = example['prediction']
        gt = example['gt']
        
        input_data = [utter] + [pred] + [gt]
        dataset = convert_input_file_to_tensor_dataset(input_data, clf_args, clf_tokenizer)

        batch = tuple(t.to('cuda') for k, t in dataset.items())

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}

            outputs = clf_model(**inputs)
            logits = outputs[0]

            logits = torch.nn.functional.softmax(logits, dim=1)

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
        
        assert len(preds) == 3, 'output must be 3 length of'
        example['empintent-utter'] = empintent_labels[int(preds[0])]
        example['empintent-pred'] = empintent_labels[int(preds[1])]
        example['empintent-gt'] = empintent_labels[int(preds[2])]

    return data

def get_empathy_emotion(data, clf_args, clf_tokenizer, clf_model, empemotion_labels):

    for example in tqdm(data):
        pred = example['prediction']
        gt = example['gt']

        input_data = [pred] + [gt]
        dataset = convert_input_file_to_tensor_dataset(input_data, clf_args, clf_tokenizer)
        
        batch = tuple(t.to('cuda') for k, t in dataset.items())

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}

            outputs = clf_model(**inputs)
            logits = outputs[0]

            logits = torch.nn.functional.softmax(logits, dim=1)

            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)

            assert len(preds) == 2, 'output must be 1 length of'

            example['empemotion-pred'] = empemotion_labels[int(preds[0])]
            example['empemotion-gt'] = empemotion_labels[int(preds[1])]
            

    return data

def get_epitome_score(data, epitome_empathy_scorer):
    pred_IP_scores, pred_EX_scores, pred_ER_scores = [], [], []
    gt_IP_scores, gt_EX_scores, gt_ER_scores = [], [], []
    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    for example in tqdm(data):
        utter = example['utterance']
        pred = example['prediction']
        gt = example['gt']
        
        pred_epitome_score = epitome_empathy_scorer([utter], [pred])
        gt_epitome_score = epitome_empathy_scorer([utter], [gt])
        
        example['epitome-IP-pred'] = int(pred_epitome_score['IP'][0][0])
        example['epitome-EX-pred'] = int(pred_epitome_score['EX'][0][0])
        example['epitome-ER-pred'] = int(pred_epitome_score['ER'][0][0])

        example['epitome-IP-gt'] = int(gt_epitome_score['IP'][0][0])
        example['epitome-EX-gt'] = int(gt_epitome_score['EX'][0][0])
        example['epitome-ER-gt'] = int(gt_epitome_score['ER'][0][0])

        pred_IP_scores += pred_epitome_score['IP'][0]
        pred_EX_scores += pred_epitome_score['EX'][0]
        pred_ER_scores += pred_epitome_score['ER'][0]
        
        gt_IP_scores += gt_epitome_score['IP'][0]
        gt_EX_scores += gt_epitome_score['EX'][0]
        gt_ER_scores += gt_epitome_score['ER'][0]

        diff_IP_scores.append(math.pow(abs(pred_epitome_score['IP'][0][0] - gt_epitome_score['IP'][0][0]), 2))
        diff_EX_scores.append(math.pow(abs(pred_epitome_score['EX'][0][0] - gt_epitome_score['EX'][0][0]), 2))
        diff_ER_scores.append(math.pow(abs(pred_epitome_score['ER'][0][0] - gt_epitome_score['ER'][0][0]), 2))
        
    return data, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    
    for line in lines:
        tokens = tokenizer.tokenize(line)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_length - special_tokens_count:
            tokens = tokens[:(args.max_seq_length - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # for interpret functions
        ref_ids = [input_ids[0]] + [pad_token_id] * len(input_ids[1:-1]) + [input_ids[-1]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    #dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    dataset = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'token_type_ids': all_token_type_ids,
    }
    
    return dataset

def get_intra_distinct(data):

    pred_dist_1, pred_dist_2 = [], []
    gold_dist_1, gold_dist_2 = [], []

    for example in data:
        pred = example['prediction']
        gold = example['gt']

        pred_m1 = InterDistinctMetric.compute(pred, 1)
        pred_m2 = InterDistinctMetric.compute(pred, 2)

        gold_m1 = InterDistinctMetric.compute(gold, 1)
        gold_m2 = InterDistinctMetric.compute(gold, 2)
        
        pred_dist_1.append(float(pred_m1))
        pred_dist_2.append(float(pred_m2))
        gold_dist_1.append(float(gold_m1))
        gold_dist_2.append(float(gold_m2))
    
    avg_pred_dist_1 = AverageMetric(sum(pred_dist_1), len(pred_dist_1))
    avg_pred_dist_2 = AverageMetric(sum(pred_dist_2), len(pred_dist_2))
    avg_gold_dist_1 = AverageMetric(sum(gold_dist_1), len(gold_dist_1))
    avg_gold_dist_2 = AverageMetric(sum(gold_dist_2), len(gold_dist_2))

    return avg_pred_dist_1, avg_pred_dist_2, avg_gold_dist_1, avg_gold_dist_2

def get_inter_distinct(data):

    pred_dist_1, pred_dist_2 = [], []
    gold_dist_1, gold_dist_2 = [], []

    for example in data:
        pred = example['prediction']
        gold = example['gt']

        pred_m1 = InterDistinctMetric.compute(pred, 1)
        pred_m2 = InterDistinctMetric.compute(pred, 2)

        gold_m1 = InterDistinctMetric.compute(gold, 1)
        gold_m2 = InterDistinctMetric.compute(gold, 2)
        
        pred_dist_1.append(float(pred_m1))
        pred_dist_2.append(float(pred_m2))
        gold_dist_1.append(float(gold_m1))
        gold_dist_2.append(float(gold_m2))
    
    avg_pred_dist_1 = AverageMetric(sum(pred_dist_1), len(pred_dist_1))
    avg_pred_dist_2 = AverageMetric(sum(pred_dist_2), len(pred_dist_2))
    avg_gold_dist_1 = AverageMetric(sum(gold_dist_1), len(gold_dist_1))
    avg_gold_dist_2 = AverageMetric(sum(gold_dist_2), len(gold_dist_2))

    return avg_pred_dist_1, avg_pred_dist_2, avg_gold_dist_1, avg_gold_dist_2


def get_classifier_args(pred_config):
    return torch.load(os.path.join(pred_config['clf_model_dir'], 'training_args.bin'))

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config['no_cuda'] else "cpu"


def load_classifier_tokenizer(args, add_prefix_space=False):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path, add_prefix_space=add_prefix_space)

def load_classifier_model(pred_config, device):
    # Check whether model exists
    if not os.path.exists(pred_config['clf_model_dir']):
        raise Exception("Model doesn't exists! Train first!")
        
    try:
        model = AutoModelForSequenceClassification.from_pretrained(pred_config['clf_model_dir']) # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
    except:
        raise Exception("Some model files might be missing...")
        
    return model

def _update_confusion_matrix(preds, golds, _report, class_lists, type):
            
    f1_dict = {}
    for class_name in class_lists:
        precisions, recalls, f1s = ConfusionMatrixMetric.compute_metrics(preds, golds, class_name)
        f1_dict[class_name] = f1s
    
        _report.update(
            {
                f'{class_name}_precision': sum(precisions, None),
                f'{class_name}_recall': sum(recalls, None),
                f'{class_name}_f1': sum(f1s, None)
            }
        )
    
    _report[f'{type}-f1'] = sum(WeightedF1Metric.compute_many(f1_dict), None)

    return _report

def get_f1_score(data):

    f1_scores = []
    for example in data:
        pred = example['prediction']
        gold = example['gt']
        
        f1_score = F1Metric.compute(pred, [gold])
        f1_scores.append(float(f1_score))
    
    return AverageMetric(sum(f1_scores), len(f1_scores))

def get_length(data):
    length_results = [len(word_tokenize(sent)) for sent in data]
    return np.average(length_results)


def evaluate_fluency(data, lm_model, lm_tokenizer):
    fluency_results = []
    for i, sent in enumerate(data):
        input_ids = lm_tokenizer(sent, return_tensors='pt').input_ids.to(device) #.input_ids.to(device)
        
        with torch.no_grad():
            outputs = lm_model(input_ids, labels=input_ids)
        
        loss = outputs.loss.item()
        ppl = math.exp(loss)

        if math.isnan(ppl):
            continue
        fluency_results.append(ppl)
    
    fluency_results = np.average(fluency_results)
    
    return fluency_results

def get_ngrams(resp, n):
    tokens = resp.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]

def get_ngram_counter(resp, n):
    ngrams = get_ngrams(resp, n)
    counter = Counter()
    counter.update(ngrams)
    return counter

def _distinct_n(data, n):
    dist_results = []
    for sent in data:
        ngram_counter = get_ngram_counter(sent.strip().lower(), n)

        if sum(ngram_counter.values()) == 0:
            print("Warning: encountered a response with no {}-grams".format(n))
            print(sent.strip().lower())
            print("ngram_counter: ", ngram_counter)
            continue
            
        dist = len(ngram_counter) / sum(ngram_counter.values())
        dist_results.append(dist)
    
    return np.average(dist_results)

def evaluate_dist_n(data, n):
    return _distinct_n(data, n)

def evaluate_acc(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    return acc

def evaluate_nidf(data, word2nidf=None):
    nidf_results = []
    for sent in data:

        avg_nidf = get_avg_nidf(sent, word2nidf)
        nidf_results.append(avg_nidf)
    
    return np.average(nidf_results)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datatype', type=str, default='test')
    parser.add_argument('--prompt_result_dir', type=str, default=None)
    parser.add_argument('--empintent_clf_dir', type=str, default=None)
    parser.add_argument('--emotion_clf_dir', type=str, default=None)
    parser.add_argument('--evaluation_save_dir', type=str, default=None)
    parser.add_argument('--wordcount_dir', type=str, default=None)
    parser.add_argument('--epitome_save_dir', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    datatype = args.datatype
    
    # load generated result from EmpGPT-3
    # you can also evaluate the generated result from Blender
    prompt_result_dir = args.prompt_result_dir
    with open(os.path.join(prompt_result_dir, 'results.pkl'), 'rb') as f:
        generation = pc.load(f)

    results = []
    for example in generation:
        utter = example['dialog'][-2]['utter']
        pred_resp = example['pred_resp'].split('The following is a conversation with an empathetic AI assistant. The assistant empathizes with human experiences and feelings well.')[0].strip().strip()
        gold_resp = example['gold_resp']
        gold_emotion = example['gold_emotion'] # for single turn
        result = {
            'utterance': utter.lower(),
            'prediction': pred_resp.lower(),
            'gt': gold_resp.lower(),
            'gt_emo': gold_emotion
        }
        results.append(result)
        
    opt = {}
    opt['no_cuda'] = False
    opt['clf_model_dir'] = args.empintent_clf_dir
    clf_args = get_classifier_args(opt)
    device = get_device(opt)
    clf_tokenizer = load_classifier_tokenizer(clf_args)
    clf_model = load_classifier_model(opt, device)
    EMPINTENTS = ['agreeing', 'acknowledging', 'encouraging', 'consoling', 'sympathizing', 'suggesting', 'questioning', 'wishing', 'neutral']
    empintent_labels = {i: empintent for i, empintent in enumerate(EMPINTENTS)}
    empintent_label2idx = {lbl: idx for idx, lbl in empintent_labels.items()}
    
    results = get_empathy_intent(results, clf_args, clf_tokenizer, clf_model, empintent_labels)
    
    EMPEMO_LABELS = ['proud', 'apprehensive', 'disappointed', 'faithful', 'impressed', 'devastated', 'prepared', 'nostalgic', 'annoyed', 'grateful', 'joyful', 'terrified', 'caring', 'trusting', 'sad', 'guilty', 'sentimental', 'hopeful', 'confident', 'surprised', 'furious', 'afraid', 'jealous', 'excited', 'lonely', 'disgusted', 'embarrassed', 'angry', 'content', 'ashamed', 'anticipating', 'anxious']
    EMPEMO_ID2LABEL = {i: emo for i, emo in enumerate(EMPEMO_LABELS)}
    emo_label2idx = {lbl: idx for idx, lbl in EMPEMO_ID2LABEL.items()}
    emo_clf_args = torch.load(os.path.join(args.emotion_clf_dir, 'training_args.bin'))
    emo_clf_tokenizer = load_classifier_tokenizer(emo_clf_args)
    opt['clf_model_dir'] = args.emotion_clf_dir
    emo_clf_model = load_classifier_model(opt, device)

    results = get_empathy_emotion(results, emo_clf_args, emo_clf_tokenizer, emo_clf_model, EMPEMO_ID2LABEL)
    
    device = 0
    opt['epitome_save_dir'] = args.epitome_save_dir
    epitome_empathy_scorer = EmpathyScorer(opt, batch_size=1, cuda_device=device)

    results, pred_IP_scores, pred_EX_scores, pred_ER_scores, gt_IP_scores, gt_EX_scores, gt_ER_scores, diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(results, epitome_empathy_scorer)

    # save evaluation result
    result_save_dir = args.evaluation_save_dir
    os.makedirs(result_save_dir, exist_ok=True)
    with open(os.path.join(result_save_dir, 'results.pkl'), 'wb') as f:
        pc.dump(results, f)

    _report = {}

    emo_preds = [example['empemotion-pred'] for example in results]
    emo_golds = [example['gt_emo'] for example in results]
    emo_acc = evaluate_acc(emo_preds, emo_golds)
    _report['EmoAcc'] = emo_acc

    _report = _update_confusion_matrix(emo_preds, emo_golds, _report, EMPEMO_LABELS, 'emotion')

    empintent_preds = [example['empintent-pred'] for example in results]
    empintent_golds = [example['empintent-gt'] for example in results]
    empintent_acc = evaluate_acc(empintent_preds, empintent_golds)

    _report['EmpIntentAcc'] = empintent_acc
    _report = _update_confusion_matrix(empintent_preds, empintent_golds, _report, EMPINTENTS, 'empintent')
    
    _report['pred_IP'] = AverageMetric(sum(pred_IP_scores), len(pred_IP_scores))
    _report['pred_EX'] = AverageMetric(sum(pred_EX_scores), len(pred_EX_scores))
    _report['pred_ER'] = AverageMetric(sum(pred_ER_scores), len(pred_ER_scores))

    _report['gt_IP'] = AverageMetric(sum(gt_IP_scores), len(gt_IP_scores))
    _report['gt_EX'] = AverageMetric(sum(gt_EX_scores), len(gt_EX_scores))
    _report['gt_ER'] = AverageMetric(sum(gt_ER_scores), len(gt_ER_scores))

    _report['diff_IP'] = AverageMetric(sum(diff_IP_scores), len(diff_IP_scores))
    _report['diff_EX'] = AverageMetric(sum(diff_EX_scores), len(diff_EX_scores))
    _report['diff_ER'] = AverageMetric(sum(diff_ER_scores), len(diff_ER_scores))
    
    word2nidf = load_word2nidf(args.wordcount_dir)

    only_pred = [example['prediction'] for example in results if example['prediction'] != '']
    only_gold = [example['gt'] for example in results]
    
    resp_typ = {
        'pred': only_pred,
        'gold': only_gold,
    }
    
    device = 'cuda'
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    lm_model.eval()

    for typ, data in resp_typ.items():
        _report[f'{typ}-PPL'] = evaluate_fluency(data, lm_model, lm_tokenizer)
        
        _report[f'{typ}-len'] = get_length(data)

        # Distinct-n
        _report[f'{typ}-dist1'] = evaluate_dist_n(data, 1)
        _report[f'{typ}-dist2'] = evaluate_dist_n(data, 2)
        _report[f'{typ}-dist3'] = evaluate_dist_n(data, 3)

        # NIDF
        _report[f'{typ}-NIDF'] = evaluate_nidf(data, word2nidf)

    f = open(os.path.join(result_save_dir, 'eval_stat.txt'), 'w')
    for k, v in _report.items():
        f.write(k + ' : ' + str(float(v)) + '\n')
    
    f.close()
