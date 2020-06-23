import torch
from tqdm import tqdm

def get_index(index, tokens, target, tokenizer):
    st, ed, idx, cnt = -1, -1, 0, 0
    for i, tok in enumerate(tokens):
        if tok == '[SEP]':
            idx = 0
            cnt += 1
            continue
        if target[idx] == tok:
            if st == -1:
                st = i
            ed = i
            idx += 1
        else:
            st, ed, idx = -1, -1, 0
        if idx >= len(target):
            break
    assert st != -1 and ed != -1
    return index[cnt]

def is_datatime(time):
    # hr:mn or hr：mn
    if time.find(':') != -1 or time.find('：') != -1:
        time = time.split(':') if time.find(':') != -1 else time.split('：')
        if len(time) != 2:
            return False
        hr, mn = time
        try:
            hr = int(hr)
            mn = int(mn)
            return True
        except:
            return False
    
    # １６時００分
    if time.find('時') != -1 and time.find('分') != -1:
        hr = time[:time.find('æ')]
        mn = time[time.find('時') + 1:time.find('分')]
        _ = time[time.find('分') + 1:]
        if _ != '':
            return False
        try:
            hr = int(hr)
            mn = int(mn)
            return True
        except:
            return False
        
    # otherwise
    return False

def post_process(
    raw_start_logits,
    raw_end_logits,
    QAExamples,
    QAFeatures,
    tokenizer,
    offset=15,
    max_len=100,
    null_threshold=0.2,
    two_ans_threshold = 0.3,
    k=5,
    special_list = ['仕様書交付期限', '入札書締切日時', '質問箇所TEL/FAX'],
    era_name = ['平成', '令和', '昭和', '大正', '明治']
):  
    # store (file_id, index, question, answer)
    answer_tuples = []
    
    for raw_start_logit, raw_end_logit, QAfeature in tqdm(zip(raw_start_logits, raw_end_logits, QAFeatures)):
        # apply softmax
        start_logit = torch.softmax(raw_start_logit, dim=0)
        end_logit = torch.softmax(raw_end_logit, dim=0)
        start_scores, start_ids = torch.topk(start_logit, k=k)
        end_scores, end_ids = torch.topk(end_logit, k=k)
        QAexample = QAExamples[QAfeature.example_id]

        # enumerate possibilities
        possible = []
        for start_score, start_idx in zip(start_scores, start_ids):
            for end_score, end_idx in zip(end_scores, end_ids):
                if start_idx == 0 or end_idx == 0 or start_idx > end_idx or end_idx - start_idx > max_len:
                    continue
                else:
                    if 0 in QAfeature.input_ids[start_idx:end_idx + 1]:
                        continue
                    possible.append((
                        start_score + end_score,
                        (start_idx, end_idx)
                    ))
        possible = sorted(possible, key=lambda t: t[0], reverse=True)

        # determine null or not
        null_score = ((start_logit[0] + end_logit[0]) * 0.5).detach().cpu().numpy()
        if len(possible) == 0 or null_score > null_threshold:
            continue
        # only one
        start_pos, end_pos = possible[0][1]
        raw_ans = tokenizer.convert_ids_to_tokens(
            QAfeature.input_ids[start_pos + offset:end_pos + offset + 1], 
            skip_special_tokens=False
        )
        ans = tokenizer.decode(
            QAfeature.input_ids[start_pos + offset:end_pos + offset + 1], 
            skip_special_tokens=False
        ).replace(' ', '').replace('#', '').replace('▁', '')
        from itertools import groupby
        raw_anss = [list(g) for k, g in groupby(raw_ans, lambda x: x == '[SEP]') if not k]
        anss = ans.split('[SEP]')
        for ans, raw_ans in zip(anss, raw_anss):
            ans = ans.replace('[PAD]', '')
            if ans == '':
                continue
            answer_tuples.append((
                QAexample.document_id,
                get_index(QAexample.index, tokenizer.tokenize(QAexample.context_text), raw_ans, tokenizer),
                QAexample.question_text,
                ans
            ))
    return answer_tuples

def _get_logit_and_null_score(raw_start_logit, raw_end_logit):
    # apply softmax
    start_logit = torch.softmax(raw_start_logit, dim=0)
    end_logit = torch.softmax(raw_end_logit, dim=0)

    # determine null or not
    null_score = ((start_logit[0] + end_logit[0]) * 0.5).detach().cpu().numpy()

    return null_score, start_logit, end_logit

def post_process_blend(
    all_raw_start_logits,
    all_raw_end_logits,
    QAExamples,
    QAFeatures,
    tokenizer,
    null_thresholds,
    offset=15,
    max_len=100,
    two_ans_threshold = 0.3,
    k=5,
    special_list = ['仕様書交付期限', '入札書締切日時', '質問箇所TEL/FAX'],
    era_name = ['平成', '令和', '昭和', '大正', '明治']
):  
    # store (file_id, index, question, answer)
    answer_tuples = []
    
    for i in range(len(all_raw_start_logits[0])):
        null_cnt, start_logits, end_logits = 0, [], []
        QAfeature = QAFeatures[i]
        for j in range(len(all_raw_start_logits)):
            raw_start_logit = all_raw_start_logits[j][i]
            raw_end_logit = all_raw_end_logits[j][i]
            null_threshold = null_thresholds[j]

            null_score, start_logit, end_logit = _get_logit_and_null_score(raw_start_logit, raw_end_logit)

            if null_score > null_threshold:
                null_cnt += 1

            start_logits.append(start_logit - null_threshold)
            end_logits.append(end_logit - null_threshold)

        if null_cnt >= (len(all_raw_start_logits) + 1) // 2:
            continue

        start_logit = sum(start_logits) / len(all_raw_start_logits)
        end_logit = sum(end_logits) / len(all_raw_start_logits)
        start_scores, start_ids = torch.topk(start_logit, k=k)
        end_scores, end_ids = torch.topk(end_logit, k=k)
        QAexample = QAExamples[QAfeature.example_id]

        # enumerate possibilities
        possible = []
        for start_score, start_idx in zip(start_scores, start_ids):
            for end_score, end_idx in zip(end_scores, end_ids):
                if start_idx == 0 or end_idx == 0 or start_idx > end_idx or end_idx - start_idx > max_len:
                    continue
                else:
                    if 0 in QAfeature.input_ids[start_idx:end_idx + 1]:
                        continue
                    possible.append((
                        start_score + end_score,
                        (start_idx, end_idx)
                    ))
        possible = sorted(possible, key=lambda t: t[0], reverse=True)

        if len(possible) == 0:
            continue

        # only one
        start_pos, end_pos = possible[0][1]
        raw_ans = tokenizer.convert_ids_to_tokens(
            QAfeature.input_ids[start_pos + offset:end_pos + offset + 1], 
            skip_special_tokens=False
        )
        ans = tokenizer.decode(
                QAfeature.input_ids[start_pos + offset:end_pos + offset + 1], 
                skip_special_tokens=False
            ).replace(' ', '').replace('#', '')
        anss = ans.split('[SEP]')
        from itertools import groupby
        raw_anss = [list(g) for k, g in groupby(raw_ans, lambda x: x == '[SEP]') if not k]
        anss = ans.split('[SEP]')
        for ans, raw_ans in zip(anss, raw_anss):
            ans = ans.replace('[PAD]', '')
            if ans == '':
                continue
            answer_tuples.append((
                QAexample.document_id,
                get_index(QAexample.index, tokenizer.tokenize(QAexample.context_text), raw_ans, tokenizer),
                QAexample.question_text,
                ans
            ))
    return answer_tuples

