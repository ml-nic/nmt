# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import json

import tensorflow as tf

from ..scripts import bleu
from ..scripts import rouge
import sys

sys.path.append("../")
import utils as our_utils
import Preprocessing.generator_utils as generator_utils

__all__ = ["evaluate"]
model_id = ""
first_fetch = True


def evaluate(ref_file, trans_file, metric, subword_option=None, question_file=None):
    """Pick a metric and evaluate depending on task."""
    # BLEU scores for translation task
    len_questions = None
    error_counter = None
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, trans_file,
                                 subword_option=subword_option)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ref_file, trans_file,
                                  subword_option=subword_option)
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, trans_file)
    elif metric.lower() == "old_accuracy":
        evaluation_score = old_accuracy(ref_file, trans_file)
    elif metric.lower() == "result_set_accuracy":
        tmp = open(ref_file, encoding="utf8").readlines()
        var_queries = False
        for s in tmp[0:10]:
            if "<var0>" in s:
                var_queries = True
        if var_queries:
            label_uri_file = "../../Data/SQA2018/dev.tok.sparql"
            evaluation_score, len_questions, error_counter = _result_set_accuracy(question_file, label_uri_file,
                                                                                  label_var_file=ref_file,
                                                                                  pred_var_file=trans_file)
            # label_uri_file, pred_uri_file=None, label_var_file=None, pred_var_file=None
        else:
            evaluation_score, len_questions, error_counter = _result_set_accuracy(question_file, ref_file, trans_file)

    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score, len_questions, error_counter


def _clean(sentence, subword_option):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()

    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)

    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
        for line in fh:
            references.append(_clean(line, subword_option))

    hypotheses = []
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(summarization_file, "rb")) as fh:
        for line in fh:
            hypotheses.append(_clean(line, subword_option=None))

    rouge_score_map = rouge.rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
    """
    Compute accuracy, each line contains a label.
    Overwritten to handle accuracy of SPARQL Queries, see func: old_accuracy, for the old version
    """
    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = re.sub(' +', ' ', label.strip())
                pred = pred_fh.readline().strip()
                pred = re.sub(' +', ' ', pred)
                pred = our_utils.rep(generator_utils.decode(pred))
                label = our_utils.rep(generator_utils.decode(label))
                if our_utils.sparql_compare(label, pred):
                    match += 1.0
                count += 1
    return 100 * match / count


# def _result_set_accuracy(question_file, label_file, pred_file):
def _result_set_accuracy(question_file, label_uri_file, pred_uri_file=None, label_var_file=None, pred_var_file=None):
    """
    Compute accuracy, each line contains a label.
    Overwritten to handle accuracy of SPARQL Queries, see func: old_accuracy, for the old version
    """
    global first_fetch
    label_json_file = "tmp/label_{}.json".format(model_id)
    pred_json_file = "tmp/pred_{}.json".format(model_id)
    count = 0
    match = 0
    if label_var_file is None:
        if pred_var_file is not None or pred_uri_file is None:
            raise Exception
    if pred_var_file is None:
        if label_var_file is not None or pred_uri_file is None:
            raise Exception
    if pred_uri_file is None:
        new_pred_uri_file = "tmp/pred_uri_{}.json".format(model_id)
        our_utils.file_wrap_replace_var_with_correct_uri(label_var_file, label_uri_file, pred_var_file,
                                                         new_pred_uri_file)
        pred_uri_file = new_pred_uri_file

    # lookup:
    # question_file, label_uri_file, pred_uri_file, label_var_file, pred_var_file

    if first_fetch is True:
        print("Only fetch data for labels one time...")
        _, _ = our_utils.fetch_results(question_file, label_uri_file, label_json_file, var_sparql_file=label_var_file)
        print("Fetched data for labels")
        first_fetch = False
    len_questions, error_counter = our_utils.fetch_results(question_file, pred_uri_file, pred_json_file,
                                                           var_sparql_file=pred_var_file)
    ground_truth = json.load(open(label_json_file))
    generated = json.load(open(pred_json_file))
    for gen_question in generated["questions"]:
        gen_nl_question = gen_question["question"][0]["string"]
        for gt_question in ground_truth["questions"]:
            gt_nl_question = gt_question["question"][0]["string"]
            if our_utils.string_compare(gen_nl_question, gt_nl_question):
                same_answer = our_utils.compare_answers(gt_question["answers"], gen_question["answers"])
                if same_answer is True:
                    match += 1
                count += 1
    return 100 * match / count, len_questions, error_counter


def old_accuracy(label_file, pred_file):
    """Compute accuracy, each line contains a label."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = label.strip().lower()
                label = re.sub(' +', ' ', label)
                pred = pred_fh.readline().strip().lower()
                pred = re.sub(' +', ' ', pred)
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
    """Compute BLEU scores using Moses multi-bleu.perl script."""

    # TODO(thangluong): perform rewrite using python
    # BPE
    if subword_option == "bpe":
        debpe_tgt_test = tgt_test + ".debpe"
        if not os.path.exists(debpe_tgt_test):
            # TODO(thangluong): not use shell=True, can be a security hazard
            subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
            subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                            shell=True)
        tgt_test = debpe_tgt_test
    elif subword_option == "spm":
        despm_tgt_test = tgt_test + ".despm"
        if not os.path.exists(despm_tgt_test):
            subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
            subprocess.call("sed s/ //g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
        tgt_test = despm_tgt_test
    cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

    # subprocess
    # TODO(thangluong): not use shell=True, can be a security hazard
    bleu_output = subprocess.check_output(cmd, shell=True)

    # extract BLEU score
    m = re.search("BLEU = (.+?),", bleu_output)
    bleu_score = float(m.group(1))

    return bleu_score
