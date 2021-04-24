#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


@get_traceback    # i.21.3.27.18:44) <-이 데코레이션때매 이함수내에서 print 출력안되는건가싶어서 잠시 코멘트아웃. ->그래도출력안되네..다시코멘트해제. /21.3.27.18:49. 
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, catId2cat):

        
    print('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj', flush=True)
    print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
    sys.stdout.flush() # i. 이것도해보고 뭐 어케해도 코랩 셀에서 출력이 안되네...;;/21.3.27.19:23.


    pq_stat = PQStat()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_id2segInfo = {segInfo['id']: segInfo for segInfo in gt_ann['segments_info']} 
        pred_id2segInfo = {segInfo['id']: segInfo for segInfo in pred_ann['segments_info']} 

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(segInfo['id'] for segInfo in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_id2segInfo: # i. 내플젝에선 이 if문 실행 안될거임. /21.3.27.17:22.
                raise KeyError("j) 내플젝에선 이것이 프린트되지 않을거임!!!!!!!") # i. 디버깅. /21.3.27.19:28.
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_id2segInfo[label]['area'] = label_cnt  # i. pred_id2segInfo[label] ex:  {'id': 0, 'category_id': 0}  (여기에 이제 'area' 도 넣어주는거지.) /21.3.27.19:33.
            pred_labels_set.remove(label)
            if pred_id2segInfo[label]['category_id'] not in catId2cat:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_id2segInfo[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_id2segInfo:
                raise KeyError("j) 내플젝에선 이것이 프린트되지 않을거임!!!!!!!") # i. 디버깅. /21.3.27.20:03.
                continue
            if pred_label not in pred_id2segInfo:
                raise KeyError("j) 내플젝에선 이것이 프린트되지 않을거임!!!!!!!") # i. 디버깅. /21.3.27.20:03.
                continue
            if gt_id2segInfo[gt_label]['iscrowd'] == 1:
                raise KeyError("j) 내플젝에선 이것이 프린트되지 않을거임!!!!!!!") # i. 디버깅. /21.3.27.20:04.
                continue
            if gt_id2segInfo[gt_label]['category_id'] != pred_id2segInfo[pred_label]['category_id']:
                continue


            # i.21.3.27.20:29) 이게 기존코드.
            # union =  gt_id2segInfo[gt_label]['area'] + pred_id2segInfo[pred_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
                                 
            # i. (1) 일단 내플젝 시각화버전(백그라운드도 foreground 클래스중 하나로 프레딕션)을 이밸류에이션하는 경우. VOID 없으니까 관련항 없애줬음.
            #  아마 기존에 이 항(gt_pred_map.get((VOID, pred_label), 0)) 을 빼줬기때매 stuff 의 PQ 가 480.9 로 100초과하는 말도안되는 값이 나왔을것으로 생각됨!! 
            #  -> 맞는듯. 이렇게 수정해주니 stuff 의 PQ 값 480.9 였던게 83.783 로 바꼈음. /21.3.27.20:23.
            union =  gt_id2segInfo[gt_label]['area'] + pred_id2segInfo[pred_label]['area'] - intersection  
            # i. TODO (2) 내플젝 이밸류에이션버전(foreground 클래스 선택지에 백그라운드 없는버전)을 이밸류에이션하는 경우,
            #  위에 코멘트아웃해둔 기존코드를 다시 사용하면 될듯. 백그라운드(VOID)를 다른카테고리로 프레딕션한건 점수깎지말아야하니까(union 에서 빼줌). /21.3.27.20:23. 

            
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_id2segInfo[gt_label]['category_id']].tp += 1
                pq_stat[gt_id2segInfo[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_id2segInfo.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_id2segInfo.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, catId2cat):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, catId2cat))
        processes.append(p)
    pq_stat = PQStat()

    for p in processes:
        pq_stat += p.get()

    # # i. pq_compute_single_core 함수 내에서 print 하는게 출력이 안돼서, 바로위 두줄(for문) 을
    # #  이렇게 바꿔서도 해봣는데, 그래도 안되네. /21.3.27.19:24.
    # import io, contextlib
    # ioJ = io.StringIO()
    # with contextlib.redirect_stdout(ioJ):
    #     for p in processes:
    #         pq_stat += p.get()
    # print(f'j) got stdout............: \n{ioJ.getvalue()}')  # i. 아무것도 출력 안됨. /21.3.27.19:02.

    return pq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    catId2cat = {cat['id']: cat for cat in gt_json['categories']} # i. 변수명 categories 였는데 내가 catId2cat 으로 바꿈. 의미명확하도록. /21.3.27.0:46.
    # i. ->참고로, pred_json 은 gt_json 의 "annotation" 만 바꿔서 만들어준거임. 따라서, 나머지 두 key 들인 "categories" 와 "images" 는 gt_json 과 동일함. /21.3.27.0:54.

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    print('j) 테스트!!! 지금이거 with contextlib.redirect_stdout(io.StringIO()): 안에서 실행되는데, 혹시 print출력된게 어딘가에서 나오나 보려고.')
    # i.21.4.22.17:51) ->이거포함, 지금 이 pq_compute 함수의 모든 print 들 죄다 코랩 출력화면(이밸류에이션결과 출력화면)에 다 프린트됨. 

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    imgId2predAnnotation = {annotation['image_id']: annotation for annotation in pred_json['annotations']} # i. 여기도 변수명 내가 바꿈. /21.3.27.0:48.
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in imgId2predAnnotation:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, imgId2predAnnotation[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, catId2cat)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(catId2cat, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results   # i.21.4.24.1:10) per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class} 
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    # print("-" * (10 + 7 * 4))
    print("-" * (10 + 10 * 4)) # i.21.4.24.2:14).

    # i.21.4.24.1:55) 바로아래에 내가 프린트하는코드 새로만들어줘서, 이건 잠시 코멘트아웃. 
    # for name, _isthing in metrics:
    #     print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
    #         name,
    #         100 * results[name]['pq'],
    #         100 * results[name]['sq'],
    #         100 * results[name]['rq'],
    #         results[name]['n'])
    #     )

    ########################################################################
    # i.21.4.24.1:15) PQ, SQ, RQ 를 모든 각각의 클래스에 대해서 출력해줘보려고 바로 위 프린트코드 복붙해서 수정해주려함. 
    #  ->잘 됨. 줄맞춤이 좀 안맞긴한데 암튼 출력될건 다 출력됨. 
    catId2cat
    per_cls_resultsJ = results['per_class']
    # i.21.4.24.1:27) 참고로 여기서 catId 에서 cat 은 COCO 형식에서의 카테고리 즉 클래스를 의미하는거임.
    #  cityscapes 에서는 'category' 라는 표현이 클래스가 아니고, 클래스를 다시 분류한, 즉 COCO형식에서의 '슈퍼카테고리'를 의미함. 
    #  정리하면, 
    #                    <클래스>     <수퍼클래스>
    #  COCO       에서는  category,   supercategory.  
    #  cityscapes 에서는  label,      category. 
    for gubunIdx, (gubunJ, printNameJ) in enumerate([('Things', 'THINGS'), ('Stuff', 'STUFF'), ('All', 'ALL')]):
        for catId, cat in catId2cat.items(): 
            # i.21.4.24.1:36) 참고로 지금 여기서 cat 은 COCO형식에서의 파놉틱세그멘테이션 gt 를 구성하는 어노json과 png 중에서
            #  어노json 의 'categories' 리스트의 한 원소 dict 임. 
            if gubunJ == 'Things':
                if cat['isthing'] != 1: continue
            elif gubunJ == 'Stuff':
                if cat['isthing'] == 1: continue
            else:
                break
            print("{:10s}| {:5.3f}  {:5.3f}  {:5.3f} {:5s}".format(
                cat['name'] if len(cat['name'])<10 else cat['name'][:10], 
                100 * per_cls_resultsJ[catId]['pq'],
                100 * per_cls_resultsJ[catId]['sq'],
                100 * per_cls_resultsJ[catId]['rq'],
                '    -') 
            )
        if gubunIdx != 2:
            print("-" * (10 + 10 * 4))
        print("{:10s}| {:5.3f}  {:5.3f}  {:5.3f} {:5d}".format(
            printNameJ,
            100 * results[gubunJ]['pq'],
            100 * results[gubunJ]['sq'],
            100 * results[gubunJ]['rq'],
            results[gubunJ]['n'])
        )
        print("-" * (10 + 10 * 4))

    ########################################################################
    
    

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
