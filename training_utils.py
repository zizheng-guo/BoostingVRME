import numpy as np
from collections import Counter
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
from sklearn.metrics import confusion_matrix
import random

def smooth(y, box_pts):
    y = [each_y for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def iou(b1, b2):
    s1, e1 = b1[0], b1[2]
    s2, e2 = b2[0], b2[2]
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union else 0.0
def nms(boxes, k_p, iou_th=0.2):
    if len(boxes) == 0:
        return boxes
    boxes = sorted(boxes, key=lambda x: x[0])
    keep = [boxes[0]]
    for b in boxes[1:]:
        if all(iou(b, k) < iou_th and abs(b[-1] - k[-1]) > k_p for k in keep):
            keep.append(b)
    return keep


def spotting(final_samples, subject_count, pred_interval, total_gt_spot, metric_final, k_p, interval_strategy, dataset_name):
    pred_subject = []
    gt_subject = []
    metric_video = MeanAveragePrecision2d(num_classes=1)
    p = 0.55 
    if dataset_name == 'SAMM_test':
        p = 0.65
    for videoIndex, video in enumerate(final_samples[subject_count]):
        preds = []
        gt = []
        score_plot = np.array(pred_interval[videoIndex])
        score_plot = smooth(score_plot, k_p * 2)
        score_plot_agg = score_plot.copy()
        thr = score_plot_agg.mean() + p * (score_plot_agg.max() - score_plot_agg.mean())
        peaks, _ = find_peaks(score_plot_agg, height=thr, distance=k_p)
        for peak in peaks:
            if interval_strategy == 1:
                n = len(score_plot_agg)
                l0 = max(0, peak - k_p)
                r0 = min(n - 1, peak + k_p)

                if dataset_name == 'SAMM_test':
                    low_threshold = 0.2
                    high_threshold = 0.8
                    n = len(score_plot_agg)
                    start = max(0, peak - 2 * k_p)
                    end = min(n - 1, peak + 2 * k_p)
                    found = False
                    near_start = max(0, peak - k_p)
                    for i in range(peak - 1, near_start - 1, -1):
                        if i > 0 and score_plot_agg[i] < low_threshold and score_plot_agg[i - 1] < low_threshold:
                            start = i
                            found = True
                            break
                    if not found:
                        far_start = max(0, peak - 2 * k_p)
                        for i in range(near_start - 1, far_start - 1, -1):
                            if i > 0 and score_plot_agg[i] < high_threshold and score_plot_agg[i - 1] < high_threshold:
                                start = i
                                break
                    found = False
                    near_end = min(n - 1, peak + k_p)
                    for i in range(peak + 1, near_end):
                        if i + 1 < n and score_plot_agg[i] < low_threshold and score_plot_agg[i + 1] < low_threshold:
                            end = i + 1
                            found = True
                            break
                    if not found:
                        far_end = min(n - 1, peak + 2 * k_p)
                        for i in range(near_end + 1, far_end):
                            if i + 1 < n and score_plot_agg[i] < high_threshold and score_plot_agg[i + 1] < high_threshold:
                                end = i + 1
                                break
                else:
                    score_plot_agg = np.array(pred_interval[videoIndex])
                    global_mean = score_plot_agg.mean()
                    threshold1 = global_mean / 1.5
                    threshold2 = global_mean
                    start = next((i - 1 for i in range(l0 + 1, peak)
                                if score_plot_agg[i] > score_plot_agg[i - 1] and
                                    score_plot_agg[i] > threshold1), l0)
                    end = next((i + 1 for i in range(r0 - 1, peak, -1)
                                if score_plot_agg[i] > score_plot_agg[i + 1] and
                                score_plot_agg[i] > threshold1), r0)
                    l_ext = max(0, peak - k_p*5)
                    r_ext = min(n - 1, peak + k_p*5)
                    for i in range(start - 1, l_ext - 1, -1):
                        if i > 0 and score_plot_agg[i] < threshold2 and score_plot_agg[i - 1] < threshold2:
                            start = i + 1
                            break
                    for j in range(end + 1, r_ext + 1):
                        if j < n - 1 and score_plot_agg[j] < threshold2 and score_plot_agg[j + 1] < threshold2:
                            end = j - 1
                            break
                preds.append([start, 0, end, 0, 0, 0, peak])
            else:
                preds.append([peak - k_p, 0, peak + k_p, 0, 0, 0, peak])

        if len(preds) == 0:
            preds = np.empty((0, 7))

        if dataset_name != 'SAMM_test':
            preds = nms(preds,k_p)

        gt_list = []
        for s in video:
            gt.append([s[0], 0, s[2], 0, 0, 0, 0, s[1]])
            gt_list.append([s[0], 0, s[2], 0, 0, 0, 0, s[1]])
            total_gt_spot += 1

        metric_video.add(np.array(preds), np.array(gt))
        metric_final.add(np.array(preds), np.array(gt))

        pred_subject.append(preds)
        gt_subject.append(gt)

    return pred_subject, gt_subject, total_gt_spot, metric_video, metric_final


def confusionMatrix(gt, pred, show=False):
    TN_recog, FP_recog, FN_recog, TP_recog = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP_recog) / (2*TP_recog + FP_recog + FN_recog)
    num_samples = len([x for x in gt if x==1])
    average_recall = TP_recog / (TP_recog + FN_recog)
    average_precision = TP_recog / (TP_recog + FP_recog)
    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall


def sequence_evaluation(total_gt_spot, metric_final): #Get TP, FP, FN for final evaluation
    TP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN_spot = total_gt_spot - TP_spot
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    return TP_spot, FP_spot, FN_spot


def mean_iou_evaluation(metric, threshold):
    all_ious = []
    for cls_df in metric.match_table:
        if not cls_df.empty:
            # 可能每格是 list/array，先全部拉平
            for cell in cls_df['iou']:
                all_ious.extend(np.array(cell).ravel())

    pos_ious = [float(iou) for iou in all_ious if float(iou) > threshold]
    return float(np.mean(pos_ious)) if pos_ious else 0.0


def convertLabel(label):
    label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    return label_dict[label]
    
def splitVideo(y1_pred, subject_count, final_samples, final_dataset_spotting): #To split y1_act_test by video
    prev=0
    y1_pred_video = []
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        y1_pred_each = y1_pred[prev:prev+len(final_dataset_spotting[countVideo+videoIndex])+1]
        y1_pred_video.append(y1_pred_each)
        prev += len(final_dataset_spotting[countVideo+videoIndex])
    return y1_pred_video

def recognition(result, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, final_samples, pred_window_list, pred_single_list, frame_skip, strategy):
    cur_pred = []
    cur_tp_gt = []
    pred_gt_recog = []
    cur_pred_window = []
    cur_pred_single = []
    pred_emotion = result  # Assuming result is the split emotion data
    pred_match_gt = sorted(metric_video.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt'].items())
    
    for video_index, video_match in pred_match_gt:
        for pred_index, sample_index in enumerate(video_match):
            pred_onset = max(0, preds[video_index][pred_index][0])
            pred_offset = max(0, preds[video_index][pred_index][2])
            start = max(0, pred_onset + 1)
            end = max(1, pred_offset - 1)
            
            # Ensure valid slice indices
            if start >= end:
                pred_emotion_list = []
            else:
                pred_emotion_list = pred_emotion[video_index][start:end]
            
            # Handle empty emotion list
            if len(pred_emotion_list) == 0:
                most_common_emotion = 4  # Default to neutral
            else:
                if strategy == 0:
                    most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
                else:
                    total = len(pred_emotion_list)
                    non_zero_list = [e for e in pred_emotion_list if e != 0]
                    non_zero_ratio = len(non_zero_list) / total if total > 0 else 0.0
                    if non_zero_ratio > 0.2:
                        counter = Counter(non_zero_list)
                        most_common_emotion, _ = counter.most_common(1)[0]
                    else:
                        counter = Counter(pred_emotion_list)
                        most_common_emotion, _ = counter.most_common(1)[0]
            
            cur_pred.append(most_common_emotion)
            pred_gt_recog.append(argmax(pred_emotion[video_index][final_samples[subject_count][video_index][0][0]]))
            gt_label = final_emotions[subject_count][video_index][sample_index]
            cur_tp_gt.append(convertLabel(gt_label) if sample_index != -1 else -1)
    
    pred_list.extend(cur_pred)
    gt_tp_list.extend(cur_tp_gt)
    pred_window_list.extend(cur_pred_window)
    pred_single_list.extend(cur_pred_single)
    print('Predicted with k_p     :', cur_pred)
    return pred_list, cur_pred, gt_tp_list, pred_window_list, pred_single_list

def recognition_evaluation(dataset_name, emotion_class, final_gt, final_pred, show=False):
    if(emotion_class == 5):
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    else:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    
    #Display recognition result
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(gt_recog, pred_recog, show)
                if(show):
                    print(emotion.title(), 'Emotion:')
                    print('TP:', TP_recog, '| FP:', FP_recog, '| FN:', FN_recog, '| TN:', TN_recog)
#                     print('Total Samples:', num_samples, '| F1-score:', round(f1_recog, 4), '| Average Recall:', round(recall_recog, 4), '| Average Precision:', round(precision_recog, 4))
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]
        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        print('------ After adding ------')
        print('TP:', TP_all, 'FP:', FP_all, 'FN:', FN_all, 'TN:', TN_all)
        print('Precision:', round(precision_all, 4), 'Recall:', round(recall_all, 4))
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4), '| F1-Score:', round(f1_all, 4))
        return f1_all
    except:
        return None


def downSampling(Y_spot, ratio):
    #Downsampling non expression samples to make ratio expression:non-expression 1:ratio

    rem_index = list(index for index, i in enumerate(Y_spot) if np.sum(i)>0)
    rem_count = int(len(rem_index) * ratio)

    #Randomly remove non expression samples (With label 0) from dataset
    if len([index for index, i in enumerate(Y_spot) if np.sum(i)==0]) <= rem_count:
        rem_count = len([index for index, i in enumerate(Y_spot) if np.sum(i)==0]) - 2
    rem_index += random.sample([index for index, i in enumerate(Y_spot) if np.sum(i)==0], rem_count) 
    rem_index.sort()
    
    # Simply return 50 index
    if len(rem_index) == 0:
        print('No index selected')
        rem_index = [i for i in range(50)]
    return rem_index