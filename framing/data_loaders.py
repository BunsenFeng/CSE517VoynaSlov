from sklearn.model_selection import train_test_split
import pickle
import os
import zipfile
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
import random
from collections import Counter
from config import PATH_TO_MFC, PATH_TO_DATA, ALL_TOPICS, FRAMES, KFOLD

# This loads all the json data in the file and returns it as a python object
# Pass it the zipfile object and the names of files to read as a list
def load_json_as_list(path, topic):
    all_text = []
    # for filename in filename_list:
    with open(os.path.join(path, topic, topic+"_labeled.json"), "r") as f:
        json_text = json.load(f)
        # no point in keeping files that aren't framing annotated
        for article_id, article in json_text.items():
            if article['annotations'].get("framing", {}) != {}:
                article['id'] = article_id
                all_text.append(article)
        # all_text += [x.update({'id':x_id}) for x_id,x in json_text.items() if x["annotations"].get("framing", {}) != {}]
    return all_text

# This converts the codes in the data json files to match the format in "codes.json"
def code_to_short_form(frame):
    if frame == "null" or frame is None:
        return None
    f = str(frame).split(".")
    return int(f[0])
    # return float(f[0] + ".0")

# This loads the codes in codes.json
def load_codes(zipfile_obj, filename):
    with zipfile_obj.open(filename) as f:
        str_to_code = json.load(f)
    return {float(k) : str_to_code[k] for k in str_to_code}

# For each doc, return text and all frames annotated in doc (by
# any annotator), and primary frame
# can give filenames or raw json as input
def format_data(data_list, issue):
    ret_list = []
    skipped = 0
    for annotated_file in data_list:
        if not "annotations" in annotated_file:
            skipped += 1
            continue
        assert "framing" in annotated_file["annotations"] and annotated_file["annotations"]["framing"] != {}

        # text = " " .join(annotated_file["text"].split()[2:])
        text = annotated_file["text"]
        if len(text) == 0:
            skipped += 1
            continue

        l = code_to_short_form(annotated_file["primary_frame"])
        # if l is None:
        #     skipped += 1
        #     continue

        # we return all frames anybody found in the document
        frames = []
        for annotation_set in annotated_file["annotations"]["framing"]:
            for frame in annotated_file["annotations"]["framing"][annotation_set]:
                frames.append({'span': (frame['start'],frame['end']), 'code': code_to_short_form(frame['code']), 'annotator':annotation_set})
        ret_list.append({"issue": issue,'id': annotated_file['id'], "text": text, "primary_frame":l, "frames":frames})

    # print("Skipped", skipped)
    return ret_list, skipped

def split_df(df, split_size):
    ids = random.sample(range(len(df)), int(len(df) * split_size))
    msk = [x in ids for x in range(len(df))]
    df1 = df[msk]
    df2 = df[[not a for a in msk]]
    return df1, df2

def split_to_topic_list(splitname):
    return ALL_TOPICS
    # if splitname in ALL_TOPICS:
    #     return [splitname]
    # else:
    #     return ALL_TOPICS

def is_between(num, start, end):
    return (num >=start) & (num <= end)

def find_sentence(sent_spans, frame_span, k=1):
    # k=1 means it's choosing one sentence with the most overlap
    overlap = {}
    frame_start, frame_end = frame_span
    for sent_idx, sent_span in enumerate(sent_spans):
        sent_start, sent_end = sent_span
        if frame_start >sent_end:
            overlap[sent_idx] = 0
        elif frame_end < sent_start:
            overlap[sent_idx]= 0
        elif is_between(frame_start, sent_start, sent_end):
            if is_between(frame_end, sent_start, sent_end):
                return [sent_idx]
            else:
                overlap[sent_idx] = sent_end-frame_start+1
        elif is_between(frame_end, sent_start, sent_end):
            overlap[sent_idx] = frame_end-sent_start+1
        elif frame_start < sent_start and sent_end < frame_end:
            overlap[sent_idx] = sent_end - sent_start+1
        else:
            print("edge case occured:")
            print("sent span", sent_span)
            print("frame span", frame_span)
            return []
    sorted_overlap = sorted(overlap.items(), key=lambda x: x[1], reverse=True)
    selected_idx = [idx for idx, o in sorted_overlap[:k] if o != 0]
    return selected_idx

from nltk.tokenize import sent_tokenize
def text_to_sentences(text, min_len=10):
    sentences = [[t for t in sent_tokenize(split)] for split in text.split("\n")]
    sentence_spans = []
    idx_start = 0
    for split in sentences:
        for sent in split:
            if len(sent) >= min_len:
                sent_span = (idx_start, idx_start+len(sent)-1)
                sentence_spans.append((sent, sent_span))
            idx_start += len(sent)
        idx_start += 1
    sentences, spans = zip(*sentence_spans)
    return sentences, spans

def aggregate_sent_labels(sent2label, id=None, filter=None):
    num_labels = []
    sent_labels = []
    for sent, labels in sent2label.items():
        num_label = len(labels)
        num_labels.append(num_label)
        if num_label == 0:
            continue

        if filter is None:
            sent_frames = list(labels.keys())
        else:
            sent_frames = []
            for label, label_annotator in labels.items():
                if len(label_annotator)>=2:
                    sent_frames.append(label)
        
        if len(sent_frames) > 0:
            sent_labels.append((sent, sent_frames, id))

        # elif num_label == 1:
        #     if filter is None:
        #         sent_frame = list(labels.keys())[0]
        #         sent_labels.append((sent, [sent_frame], id))
        #     else:
        #         continue
        # else:
        #     sent_frames = []
        #     for label, label_annotator in labels.items():
        #         if len(label_annotator)>=2:
        #             sent_frames.append(label)
        #     if len(sent_frames) > 0:
        #         sent_labels.append((sent, sent_frames, id))
        
    return sent_labels, num_labels        


def convert_to_sentence(topic_data, filter=None):
    data = []
    num_labels = Counter()

    for article in topic_data:
        sentences, sentence_spans = text_to_sentences(article['text'])
        sent2label = {s:{} for s in sentences}
        for frame in article['frames']:
            frame_code, frame_span, frame_annotator = frame['code'], frame['span'], frame['annotator']
            matched_sentence_idxs = find_sentence(sentence_spans, frame_span, k=1)
            if len(matched_sentence_idxs) == 0:
                # no sentence matched to the frame span
                continue
            for matched_sent_idx in matched_sentence_idxs:
                matched_sentence = sentences[matched_sent_idx]
                if frame_code not in sent2label[matched_sentence]:
                    sent2label[matched_sentence][frame_code] = set()
                sent2label[matched_sentence][frame_code].add(frame_annotator)
        data_article, _num_labels = aggregate_sent_labels(sent2label, article['id'], filter=filter)
        data.extend(data_article)
        num_labels.update(_num_labels)
    # print(num_labels)
    return data

def print_label_states(data):
    # average number of label + label distribution
    label_counter = Counter()
    num_labels = []
    for sentence_labels in data['test']['label']:
        label_counter.update(sentence_labels)
        num_labels.append(len(sentence_labels))
    total = sum(label_counter.values())
    sorted_counts = sorted(list(label_counter.items()), key=lambda v: int(v[0]))
    print([(FRAMES[str(int(k))],round(v/total*100,2))  for k,v in sorted_counts])
    print(round(sum(num_labels)/len(num_labels),2))

def foldidx_to_splitidxs(idx, num_fold):
    test_idx = idx
    dev_idx = (idx+1) % num_fold
    train_idxs = [i for i in range(num_fold) if i not in [test_idx, dev_idx]]
    return train_idxs, dev_idx, test_idx

def get_data_from_foldidx(data_by_fold, fold_idx, k, no_test=False):
    train_idxs, dev_idx, test_idx = foldidx_to_splitidxs(fold_idx, k)
    test_list = data_by_fold[test_idx]
    dev_list = data_by_fold[dev_idx]
    train_list = []
    for train_idx in train_idxs:
        train_list.extend(data_by_fold[train_idx])
    if no_test:
        train_list.extend(test_list)
        return train_list, dev_list, None
    return train_list, dev_list, test_list

def split_data(data, k, no_test=False):
    data_by_fold = divide_data_by_fold(data, KFOLD)
    train, dev, test = get_data_from_foldidx(data_by_fold, k, KFOLD, no_test)
    train = pd.DataFrame(train,columns = ["sentence","label","id"])
    dev = pd.DataFrame(dev,columns = ["sentence","label","id"])
    if not no_test:
        test = pd.DataFrame(test,columns = ["sentence","label","id"])
    return train, dev, test


def divide_data_by_fold(samples, num_fold):
    random.seed(11)
    random.shuffle(samples)
    num_each_fold = int(len(samples)/num_fold)
    fold_samples = {}
    for i in range(num_fold):
        _fold_samples = samples[i*num_each_fold:(i+1)*num_each_fold]
        fold_samples[i] = _fold_samples
    fold_samples[num_fold-1] += samples[num_fold*num_each_fold:]
    return fold_samples

def load_annotated_voynaslov_data():
    from config import PATH_TO_ANNOTATED_VOYNASLOV, FRAME2NUM
    # TODO
    data = pd.read_csv(PATH_TO_ANNOTATED_VOYNASLOV, sep="\t")
    labels = []
    for _, row in data.iterrows():
        _labels = row['labels_frame'].split("|")
        _labelidxs = [FRAME2NUM[l] for l in _labels]
        labels.append((row['translation'], _labelidxs, "voynaslov-"+str(row['id'])))
    return labels

def load_mfc_data(splitname, k, filter=None, force_reset = False, bool_fewshot=False):
    if "_filter" in splitname:
        filter=True
        splitname = splitname.replace("_filtered","")
    
    split_index = splitname
    if filter is not None:
        split_index = split_index+"_filtered"
    if bool_fewshot:
        split_index = split_index+"_fewshot"
    split_index += "_"+str(k)

    print("Loading Data:", split_index)
    data_path = os.path.join(PATH_TO_DATA, split_index)
    if os.path.exists(data_path) and not force_reset:
        data =  load_from_disk(data_path)
        try:
            indomains = load_from_disk(data_path+"-indomain")
        except:
            indomains = None
        print(split_index)
        print(f"*Size - total:{len(data['train'])+len(data['dev'])+len(data['test'])}, train:{len(data['train'])}, dev: {len(data['dev'])}, test: {len(data['test'])}")
        print_label_states(data)
        print("")
        return data, indomains

    data_by_topic = {}
    topics = split_to_topic_list(splitname)
    for topic in topics:
        data_by_topic[topic], num_skipped = format_data(load_json_as_list(PATH_TO_MFC, topic), topic)
        # print(topic, len(data_by_topic[topic]), "skipped: ", num_skipped)

    # randomly print out data
    for topic in data_by_topic:
        x = random.choice(data_by_topic[topic])
        for frame in x['frames'][-2:]:
            print(f"[{x['primary_frame']}, {frame['code']}] span:{x['text'][frame['span'][0]:frame['span'][1]]}")
        break

    # convert to sentence level labels
    data_by_sentence = {}
    for topic, topic_data in data_by_topic.items():
        data_by_sentence[topic] = convert_to_sentence(topic_data, filter=filter)


    if splitname == "random_stratified":
        train_df, test_df, dev_df = None, None, None
        for topic, topic_data in data_by_sentence.items():
            _train_df, _dev_df, _test_df = split_data(topic_data, k)
            # df = pd.DataFrame(topic_data).sample(frac=1)
            # df.columns = ["sentence","label","id"]
            # _train_df, dev_test_df = train_test_split(df, test_size=0.2)
            # _dev_df, _test_df = train_test_split(dev_test_df, test_size=0.5)
            if train_df is None:
                train_df, test_df, dev_df = _train_df, _test_df, _dev_df
            else:
                train_df = pd.concat([train_df, _train_df])
                test_df = pd.concat([test_df, _test_df])
                dev_df = pd.concat([dev_df, _dev_df])
    elif splitname == "random":
        all_data = [d for topic, topic_data in data_by_sentence.items() for d in topic_data]
        train_df, dev_df, test_df = split_data(all_data, k)
        # df = pd.DataFrame(all_data).sample(frac=1)
        # df.columns = ["sentence","label","id"]
        # train_df, dev_test_df = train_test_split(df, test_size=0.2)
        # dev_df, test_df = train_test_split(dev_test_df, test_size=0.5)

    elif splitname in ALL_TOPICS:
        topicname = splitname
        assert topicname in data_by_sentence
        train_data = [d for topic, topic_data in data_by_sentence.items() for d in topic_data if topic != topicname]
        train_df, dev_df, _t = split_data(train_data, k, no_test=True)
        assert _t is None
        # df = pd.DataFrame(train_data).sample(frac=1)
        # df.columns = ["sentence","label","id"]
        # train_df, dev_df = train_test_split(df, test_size=0.1)
        test_data = [d for d in data_by_sentence[topicname]]
        test_df = pd.DataFrame(test_data, columns=["sentence","label","id"]).sample(frac=1)

    elif splitname == "voynaslov":
        data_voynaslov = load_annotated_voynaslov_data()
        random.shuffle(data_voynaslov)
        data_by_fold = divide_data_by_fold(data_voynaslov, KFOLD)

        train_list = []
        for train_idx in [0,1,2,3,4]:
            train_list.extend(data_by_fold[train_idx])
        dev_list = []
        for dev_idx in [5,6,7,8]:
            dev_list.extend(data_by_fold[dev_idx])
        test_list = []
        for test_idx in [9]:
            test_list.extend(data_by_fold[test_idx])
        train_df = pd.DataFrame(train_list,columns = ["sentence","label","id"])
        dev_df = pd.DataFrame(dev_list,columns = ["sentence","label","id"])
        test_df = pd.DataFrame(test_list,columns = ["sentence","label","id"])    
    if bool_fewshot:
        print(splitname)
        data_voynaslov = load_annotated_voynaslov_data()
        random.shuffle(data_voynaslov)
        # _train_df, _dev_df, _test_df = split_data(data_voynaslov, k)
        data_by_fold = divide_data_by_fold(data_voynaslov, KFOLD)

        train_list = []
        for train_idx in [0,1,2,3,4]:
            train_list.extend(data_by_fold[train_idx])
        dev_list = data_by_fold[5]
        test_list = []
        for test_idx in [6,7,8,9]:
            test_list.extend(data_by_fold[test_idx])
        # train, dev, test = get_data_from_foldidx(data_by_fold, k, KFOLD, no_test)
        train_df = pd.DataFrame(train_list,columns = ["sentence","label","id"])
        dev_df = pd.DataFrame(dev_list,columns = ["sentence","label","id"])
        test_df = pd.DataFrame(test_list,columns = ["sentence","label","id"])
        # _train_df.to_csv(f"./annotated/final_annotation_train_{k}.tsv", sep="\t")
        # _dev_df.to_csv(f"./annotated/final_annotation_dev_{k}.tsv", sep="\t")
        # _test_df.to_csv(f"./annotated/final_annotation_test_{k}.tsv", sep="\t")
        # train_df = pd.concat([train_df, _train_df])
        # test_df = pd.concat([test_df, _test_df])
        # dev_df = pd.concat([dev_df, _dev_df])

    # data = {"train":train_df, "test":test_df, "dev":dev_df}
    train = Dataset.from_pandas(train_df, split="train")
    test = Dataset.from_pandas(test_df, split="test")
    dev = Dataset.from_pandas(dev_df, split="dev")
    data = DatasetDict({"train": train, "test": test, "dev": dev})
    
    indomains = None
    if bool_fewshot:
        train = Dataset.from_pandas(_train_df, split="test")
        test = Dataset.from_pandas(_test_df, split="test")
        dev = Dataset.from_pandas(_dev_df, split="dev")
        indomains = DatasetDict({"train":train, "test": test, "dev": dev})
        indomains.save_to_disk(data_path+"-indomain")
    print(f"*Size - total:{len(data['train'])+len(data['dev'])+len(data['test'])}, train:{len(data['train'])}, dev: {len(data['dev'])}, test: {len(data['test'])}")

    data.save_to_disk(data_path)
    print("processed data saved to ", data_path)
    return data, indomains


def main():
    # for k in range(KFOLD):
    for k in [0]:
        df = load_mfc_data("random_stratified", k)
        for topic in ALL_TOPICS:
            df = load_mfc_data(topic, k)

        df = load_mfc_data("random_stratified", filter=True, k=k)
        for topic in ALL_TOPICS:
            df = load_mfc_data(topic, filter=True, k=k)

if __name__ == "__main__":
    main()