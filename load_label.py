import copy

def load_label(dataset_name, images, subjects, subjectsVideos, codeFinal, frame_skip):
    # Filter the emotion to be evaluated
    emotion_list = ['repression', 'anger', 'contempt', 'disgust', 'fear', 'sadness', 'sad', 'negative', 'happy', 'happiness', 'positive', 'surprise', 'others', 'other']
    emotion_negative = ['negative']
    emotion_positive = ['happiness']
    emotion_surprise = ['surprise']
    emotion_others = []

    if dataset_name == 'CASME_3' or  dataset_name == 'CASME_test':
        emotion_positive = ['happy']
        emotion_negative = ['anger', 'disgust', 'fear', 'sad']
        emotion_others = ['others']
    elif(dataset_name == 'SAMMLV'): 
        emotion_negative = ['anger', 'contempt', 'disgust', 'fear', 'sadness']
        emotion_others = ['other']

        
    vid_need_spot = [] #Micro or Macro
    vid_need_recog = [] #Emotion match with emotion_type
    vid_count = 0
    ground_truth = []
    emotions = []
    only_spot = 0
    for sub_video_each_index, sub_vid_each in enumerate(subjectsVideos):
        ground_truth.append([])
        emotions.append([])
        for videoIndex, videoCode in enumerate(sub_vid_each):
            each_emotion = []
            on_off = []
            for i, row in codeFinal.iterrows():
                if (str(row['subjectCode']).split('.')[-1]==str(subjects[sub_video_each_index])): #S15, S16... for CAS(ME)^2, 001, 002... for SAMMLV
                    if (row['videoCode']==videoCode):
                        if  (row['type']=='micro-expression'): #Micro-expression or Macro-expression
                            vid_need_spot.append(vid_count) #To get the video that is needed
                            if only_spot == 0:
                                row['emotion'] = row['emotion'].lower()
                            if only_spot or (row['emotion'] in emotion_list): #check emotion
                                if only_spot:
                                    each_emotion.append('others')
                                else:
                                    if (dataset_name == 'CASME2' or dataset_name == 'CASME_sq'):
                                        each_emotion.append(row['emotion'])
                                    else:
                                        if (row['emotion'] in emotion_negative):
                                            each_emotion.append('negative')
                                        if (row['emotion'] in emotion_positive):
                                            each_emotion.append('positive')
                                        if (row['emotion'] in emotion_surprise):
                                            each_emotion.append('surprise')
                                        if(row['emotion'] in emotion_others):
                                            each_emotion.append('others')
                                if (row['offset']==0): #Take apex if offset is 0
                                    on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['apex']-1)]) 
                                else:
                                    if(dataset_name=='CASME2'):
                                        if(isinstance(row['apex'], int)): # CASME2 has one sample has '/' at apex frame
                                            if([int(row['onset']-1), int(row['apex'])-1, int(row['offset'])-1] != [65, 110, 110]): # CASME2 has one sample has index error
                                                on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['offset'])-1])
                                    elif(int(row['onset'])!=0): # Ignore the samples that is extremely long in SAMMLV, suggested in other papers
                                        on_off.append([int(row['onset']-1), int(row['apex']-1), int(row['offset']-1)])

            #To get the video that is needed for recognition
            if(len(each_emotion)>0 and len(on_off)>0):
                vid_need_recog.append(vid_count) 
            ground_truth[-1].append(on_off) 
            emotions[-1].append(each_emotion)
            vid_count+=1
        
    #Remove unused video
    if images is None:
        final_images = None
    else:
        final_images = [images[i] for i in vid_need_recog]
    final_samples = []
    final_videos = []
    final_subjects = []
    final_emotions = []
    # final_emotions_extend = []
    count = 0
    for subjectIndex, subject in enumerate(ground_truth):
        final_samples.append([])
        final_videos.append([])
        final_emotions.append([])
        for samplesIndex, samples in enumerate(subject):
            if (count in vid_need_recog):
                final_samples[-1].append(samples)
                final_videos[-1].append(subjectsVideos[subjectIndex][samplesIndex])
                final_subjects.append(subjects[subjectIndex])
                final_emotions[-1].append(emotions[subjectIndex][samplesIndex])
                # final_emotions_extend.extend(emotions[subjectIndex][samplesIndex])
            count += 1

    #Remove the empty data in array
    final_subjects = list(dict.fromkeys(final_subjects))
    final_videos = [ele for ele in final_videos if ele != []]
    final_samples = [ele for ele in final_samples if ele != []]
    final_emotions = [ele for ele in final_emotions if ele != []]

    final_samples_labeling = copy.deepcopy(final_samples)
    for subject_index, subject in enumerate(final_samples_labeling):
        for video_index, video in enumerate(subject):
            for sample_index, sample in enumerate(video):
                for phase_index, phase in enumerate(sample):
                    final_samples_labeling[subject_index][video_index][sample_index][phase_index] = int(phase/frame_skip)

    print('Final Ground Truth Data')
    print('Subjects Name', final_subjects)
    print('Videos Name: ', final_videos)
    print('Samples [Onset, Apex, Offset]: ', final_samples_labeling)
    print('Emotions:', final_emotions)
    # print('Total Videos:', len(final_images))

    return final_images, final_subjects, final_videos, final_samples_labeling, final_emotions


# Calculate k', average length of half of an expression
def cal_k_p(dataset_name, final_samples):
    if dataset_name == "SAMM_test" or dataset_name == "CASME_test":
        return 6
    samples = [samples for subjects in final_samples for videos in subjects for samples in videos]
    
    # total_duration = 0
    # for sample in samples:
    #     total_duration += sample[2]-sample[0] #min(50, sample[2]-sample[0])
    # N=total_duration/len(samples)
    # k_p=int((N+1)/2)

    durations = [sample[2] - sample[0] for sample in samples]
    durations.sort()
    median_duration = durations[len(durations) // 2]
    N = median_duration
    k_p = int((N + 1) / 2)

    print(dataset_name, 'k_p =', k_p)
    return k_p