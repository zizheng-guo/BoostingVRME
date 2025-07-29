import numpy as np
import cv2
import dlib
import time
import scipy.fftpack as fftpack


def feature_extraction_spotting(dataset_name, final_images, k):
    detector = dlib.get_frontal_face_detector()  
    predictor = dlib.shape_predictor('Utils/shape_predictor_68_face_landmarks.dat') 
    print('Running')
    start = time.time()
    dataset = []
    for video in range(len(final_images)):
        OFF_video = []
        for img_count in range(final_images[video].shape[0] // (k//2) -1):
            final_image = flowProcess(detector, predictor, final_images[video], img_count * (k//2), k, True, dataset_name, 320, 1)
            OFF_video.append(final_image)

        dataset.append(OFF_video)   
        print('Video', video, 'Done')
    print('All Done')
    end = time.time()
    print('Total time taken: ' + str(end-start) + 's')
    return dataset


def crop_picture(detector, predictor, img_rd, size, dataset_name):
    img_gray = img_rd
    faces = detector(img_gray, 0)
    if len(faces) == 0:
        return None, None, 0, 0, 0, 0
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])

    if len(faces) == 0:
        return None, None, 0, 0, 0, 0
    
    #两个眼角的位置
    left=landmarks[39]
    right=landmarks[42]

    gezi=int((right[0,0]-left[0,0])/2)
    center=[int((right[0,0]+left[0,0])/2),int((right[0,1]+left[0,1])/2)]

    b=center[1] +int(5.5 * gezi)
    d=center[0] +int(4.5 * gezi)
    a=max((center[1] - int(3.5 * gezi)),0)
    c=max(center[0] - int(4.5 * gezi),0)

    img_crop = img_rd[a:b, c:d]
    img_crop_samesize = cv2.resize(img_crop, (size, size))
    return landmarks, img_crop_samesize, a, b, c, d


def get_roi_bound(low, high, bia, landmark0):
    roi1_points = landmark0[low:high]

    roi1_high = roi1_points[:, 0].argmax(axis=0)
    roi1_low = roi1_points[:, 0].argmin(axis=0)
    roi1_left = roi1_points[:, 1].argmin(axis=0)
    roil_right = roi1_points[:, 1].argmax(axis=0)

    roil_h = roi1_points[roi1_high, 0]
    roi1_lo = roi1_points[roi1_low, 0]
    roi1_le = roi1_points[roi1_left, 1]
    roil_r = roi1_points[roil_right, 1]

    roil_h_ex = (roil_h + bia)[0, 0]
    roi1_lo_ex = (roi1_lo - bia)[0, 0]
    roi1_le_ex = (roi1_le - bia)[0, 0]
    roil_r_ex = (roil_r + bia)[0, 0]
    return (roil_h_ex), (roi1_lo_ex), (roi1_le_ex), (roil_r_ex)
 

def get_roi(flow, percent):
    if flow.size == 0:
        return 0,0
    r1, theta1 = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    r1 = np.ravel(r1)

    x1 = np.ravel(flow[:, :, 0])
    y1 = np.ravel(flow[:, :, 1])
    arg = np.argsort(r1)  # 代表了r1这个矩阵内元素的从小到大顺序
    num = int(len(r1) * (1 - percent))
    x_new = 0
    y_new = 0

    for i in range(num, len(arg)):  # 取相对比较大的
        a = arg[i]
        x_new += x1[a]
        y_new += y1[a]
    x = x_new / (len(arg) - num)
    y = y_new / (len(arg) - num)
    return x, y


def tu_landmarks(detector, predictor, gray, img_rd, landmark0, frame_shang, frame_left, h, w, img_size):
    try:
        faces = detector(gray, 0)
    except:
        return None
    if (len(faces) == 0):
        landmark0[:, 0] = (landmark0[:, 0] - frame_left) * (img_size / w)
        landmark0[:, 1] = (landmark0[:, 1] - frame_shang) * (img_size / h)
        landmarkss = landmark0
    else:
        landmarkss = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])
    return landmarkss


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    fft = fftpack.fft(tensor, axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[bound_high:-bound_high] = 0
    iff = fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def flowProcess(detector, predictor, img, start, interval, align, dataset_name, img_size, fs):
    flow = []
    flow1_total = [[0, 0]] 
    flow1_total1 = [[0, 0]]
    flow1_total2 = [[0, 0]]
    flow1_total3 = [[0, 0]]
    flow3_total = [[0, 0]]
    flow3_total1 = [[0, 0]]
    flow3_total2 = [[0, 0]]
    flow3_total3 = [[0, 0]]
    flow4_total = [[0, 0]]
    flow4_total1 = [[0, 0]]
    flow4_total2 = [[0, 0]]
    flow4_total3 = [[0, 0]]
    flow4_total4 = [[0, 0]]
    flow4_total5 = [[0, 0]]
    flow5_total1 = [[0, 0]]
    flow5_total2 = [[0, 0]]
    flow6_total= [[0, 0]]
    flow7_total= [[0, 0]]
    i = -1
    flag = 0
    while i < interval - 1:
        i += 1
        frame = start + i

        if flag == 0:
            img_rd = img[frame] 
            landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(detector, predictor, img_rd, img_size, dataset_name)

            if landmark0 is None:
                # frame_detect = frame
                # while landmark0 is None and frame_detect > 0:
                #     frame_detect -= 3
                #     landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(detector, predictor, img[frame_detect], img_size, dataset_name)
                # frame_detect = frame
                # while landmark0 is None and frame_detect < len(img):
                #     frame_detect += 3
                #     landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(detector, predictor, img[frame_detect], img_size, dataset_name)
                print("Warning: face detection failed",frame)
                if i == interval -1:
                    continue
                flow1_total.append([0, 0])
                flow1_total1.append([0, 0])  
                flow1_total2.append([0, 0])
                flow1_total3.append([0, 0])
                flow3_total.append([0, 0])
                flow3_total1.append([0, 0])  
                flow3_total2.append([0, 0])
                flow3_total3.append([0, 0])
                flow4_total.append([0, 0])
                flow4_total1.append([0, 0])  
                flow4_total2.append([0, 0])
                flow4_total3.append([0, 0])
                flow4_total4.append([0, 0])
                flow4_total5.append([0, 0])
                flow5_total1.append([0, 0])
                flow5_total2.append([0, 0])  
                flow6_total.append([0, 0])
                flow7_total.append([0, 0])   
                # interval = interval - 1             
                continue


            # 记录框的位置，上下左右在整个图片中的坐标，和68点的位置。img_rd是被裁减之后的面部位置，并resize
            gray = img_rd 
            landmark0 = tu_landmarks(detector, predictor, gray, img_rd, landmark0, frame_shang, frame_left, frame_xia - frame_shang, frame_right - frame_left, img_size)  # 对人脸68个点的定位

            # 相对与新图片的68点的位置, 左眉毛的位置
            roi1_right, roi1_left, roi1_low, roi1_high = get_roi_bound(17, 22, 0, landmark0) 
            roi1_sma = []  # 存储了左眼的三个小的感兴趣区域，从里到外
            roi1_sma.append([landmark0[20, 1] - (roi1_low - 15), landmark0[20, 0] - (roi1_left-5)])
            roi1_sma.append([landmark0[19, 1] - (roi1_low - 15), landmark0[19, 0] - (roi1_left-5)])
            roi1_sma.append([landmark0[18, 1] - (roi1_low - 15), landmark0[18, 0] - (roi1_left-5)])
            # cv2.rectangle(img_rd, (landmark0[20, 0] - 10, landmark0[20, 1] + 10),
            #               (landmark0[20, 0] + 10, landmark0[20, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[19, 0] - 10, landmark0[19, 1] + 10),
            #               (landmark0[19, 0] + 10, landmark0[19, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[18, 0] - 10, landmark0[18, 1] + 10),
            #               (landmark0[18, 0] + 10, landmark0[18, 1] - 10), (0, 255, 255), 1)
            prevgray_roi1 = gray[max(0,roi1_low - 15):roi1_high + 5, max(0,roi1_left - 5):roi1_right]

            # 右眼
            roi3_right, roi3_left, roi3_low, roi3_high = get_roi_bound(22, 27, 0, landmark0)
            roi3_sma = []  # 存储了右眼的三个小的感兴趣区域，从里到外
            roi3_sma.append([landmark0[23, 1] - (roi3_low - 15), landmark0[23, 0] - roi3_left])
            roi3_sma.append([landmark0[24, 1] - (roi3_low - 15), landmark0[24, 0] - roi3_left])
            roi3_sma.append([landmark0[25, 1] - (roi3_low - 15), landmark0[25, 0] - roi3_left])
            # cv2.rectangle(img_rd, (landmark0[25, 0] - 10, landmark0[25, 1] + 10),
            #               (landmark0[25, 0] + 10, landmark0[25, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[24, 0] - 10, landmark0[24, 1] + 10),
            #               (landmark0[24, 0] + 10, landmark0[24, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[23, 0] - 10, landmark0[23, 1] + 10),
            #               (landmark0[23, 0] + 10, landmark0[23, 1] - 10), (0, 255, 255), 1)
            prevgray_roi3 = gray[max(0,roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]

            # 嘴巴处的四个
            roi4_right, roi4_left, roi4_low, roi4_high = get_roi_bound(48, 67, 0, landmark0)
            roi4_sma = []
            roi4_sma.append([landmark0[48, 1] - (roi4_low - 15), landmark0[48, 0] - (roi4_left - 20)])
            roi4_sma.append([landmark0[54, 1] - (roi4_low - 15), landmark0[54, 0] - (roi4_left - 20)])
            roi4_sma.append([landmark0[51, 1] - (roi4_low - 15), landmark0[51, 0] - (roi4_left - 20)])
            roi4_sma.append([landmark0[57, 1] - (roi4_low - 15), landmark0[57, 0] - (roi4_left - 20)])
            roi4_sma.append([landmark0[62, 1] - (roi4_low - 15), landmark0[62, 0] - (roi4_left - 20)])
            # cv2.rectangle(img_rd, (landmark0[48, 0] -10, landmark0[48, 1] + 10),
            #               (landmark0[48, 0] + 10, landmark0[48, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[51, 0] - 10, landmark0[51, 1] + 10),
            #               (landmark0[51, 0] + 10, landmark0[51, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[54, 0] - 10, landmark0[54, 1] + 10),
            #               (landmark0[54, 0] + 10, landmark0[54, 1] - 10), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[57, 0] - 10, landmark0[57, 1] + 10),
            #               (landmark0[57, 0] + 10, landmark0[57, 1] - 10), (0, 255, 255), 1)
            prevgray_roi4 = gray[roi4_low - 15:roi4_high + 10, max(0, roi4_left - 20):roi4_right + 20]

            # 鼻子两侧
            roi5_right, roi5_left, roi5_low, roi5_high = get_roi_bound(30, 36, 0, landmark0)
            roi5_sma = []
            roi5_sma.append([landmark0[31, 1] - (roi5_low - 20), landmark0[31, 0] - (roi5_left - 30)])
            roi5_sma.append([landmark0[35, 1] - (roi5_low - 20), landmark0[35, 0] - (roi5_left - 30)])
            # cv2.rectangle(img_rd, (landmark0[31, 0] -20, landmark0[31, 1] + 5),
            #               (landmark0[31, 0] + 10, landmark0[31, 1] - 20), (0, 255, 255), 1)
            # cv2.rectangle(img_rd, (landmark0[35, 0] - 10, landmark0[35, 1] + 5),
            #               (landmark0[35, 0] + 20, landmark0[35, 1] - 20), (0, 255, 255), 1)
            prevgray_roi5 = gray[roi5_low - 20:roi5_high + 5, max(0, roi5_left - 30):roi5_right + 30]



            #左眼睑部位
            roi6_right, roi6_left, roi6_low, roi6_high = get_roi_bound(36, 42, 0, landmark0)
            width=roi6_right-roi6_left
            height=width/2
            xin=(roi6_high+ roi6_low)/2
            roi6_high=int(xin+3*height/2)
            roi6_low=int(xin+height/2)
            roi6_left = max(0, roi6_left)
            prevgray_roi6 = gray[roi6_low :roi6_high, roi6_left :roi6_right]
            # cv2.rectangle(img_rd, (roi6_left, roi6_high ), (roi6_right, roi6_low ), (0, 255, 255), 1)

            # 右眼睑部位
            roi7_right, roi7_left, roi7_low, roi7_high = get_roi_bound(42, 48, 0, landmark0)
            width = roi7_right - roi7_left
            height = width / 2
            xin = (roi7_high + roi7_low) / 2
            roi7_high = int(xin + 3 * height / 2)
            roi7_low = int(xin + height / 2)
            roi7_left = max(0, roi7_left)
            prevgray_roi7 = gray[roi7_low :roi7_high , roi7_left :roi7_right]
            # cv2.rectangle(img_rd, (roi7_left, roi7_high), (roi7_right, roi7_low), (0, 255, 255), 1)


            # 全局
            roi2_right, roi2_left, roi2_low, roi2_high = get_roi_bound(29, 31, 13, landmark0)
            prevgray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]    

            roi1_sma = np.array(roi1_sma)
            roi3_sma = np.array(roi3_sma)
            roi4_sma = np.array(roi4_sma)
            roi5_sma = np.array(roi5_sma)

            # cv2.imwrite("output.jpg", img_rd)
            # cv2.imwrite("output.jpg", img_rd)
            flag = 1

        else:
            img_rd1 = img[frame] 
            img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]
            img_rd = cv2.resize(img_crop, (img_size, img_size))
            gray = img_rd
            bia = 10

            gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
            flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5, 0) 
            flow2 = np.array(flow2)
            x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)

            # 面部对齐，移动切割框
            if align == True:
                l = 0
                while ((x1 ** 2 + y1 ** 2) > 1.0):  # 移动比较大，相应移动脸的位置
                    l = l + 1
                    if (l > 3):
                        print("l3")
                        break
                    frame_left += int(round(x1))
                    if frame_left < 0:
                        frame_left = 0
                    frame_shang += int(round(y1))
                    if frame_shang < 0:
                        frame_shang = 0
                    frame_right += int(round(x1))
                    frame_xia += int(round(y1))
                    img_rd1 = img[frame]
                    img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]
                    img_rd = cv2.resize(img_crop, (img_size, img_size))
                    gray = img_rd
                    # 求全局的光流
                    gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                    flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5,0)
                    flow2 = np.array(flow2)
                    x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)

            gray_roi1 = gray[max(0,roi1_low - 15):roi1_high + 5, max(0,roi1_left - 5):roi1_right]
            flow1 = cv2.calcOpticalFlowFarneback(prevgray_roi1, gray_roi1, None, 0.5, 3, 15, 5, 7, 1.5,0) 
            bia_top = max(0, bia + min(0,roi1_low - 15))
            bia_low = bia
            if roi1_high + 5 - max(0,roi1_low - 15) - 3 < bia + bia_top:
                bia_low = 4
        
            a, b = get_roi(flow1[bia_top:-bia_low, bia:-bia, :], 0.3)  # 去掉光流特征矩阵周边round大小的部分，求均值
            a1, b1 = get_roi(flow1[roi1_sma[0, 0] - 10:roi1_sma[0, 0] + 10, roi1_sma[0, 1] - 10:roi1_sma[0, 1] + 10, :],0.2)
            a2, b2 = get_roi(flow1[roi1_sma[1, 0] - 10:roi1_sma[1, 0] + 10, roi1_sma[1, 1] - 10:roi1_sma[1, 1] + 10, :],0.2)
            a3, b3 = get_roi(flow1[roi1_sma[2, 0] - 10:roi1_sma[2, 0] + 10, roi1_sma[2, 1] - 10:roi1_sma[2, 1] + 10, :],0.2)
            flow1_total1.append([a1 - x1, b1 - y1])  
            flow1_total2.append([a2 - x1, b2 - y1])
            flow1_total3.append([a3 - x1, b3 - y1])
            flow1_total.append([a - x1, b - y1])

            gray_roi3 = gray[max(0,roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
            flow3 = cv2.calcOpticalFlowFarneback(prevgray_roi3, gray_roi3, None, 0.5, 3, 15, 5, 7, 1.5, 0) 
            bia_top = max(0, bia + min(0,roi3_low - 15))
            a, b = get_roi(flow3[bia_top:-bia, bia:-bia, :], 0.3)
            a1, b1 = get_roi(flow3[roi3_sma[0, 0] - 10:roi3_sma[0, 0] + 10, roi3_sma[0, 1] - 10:roi3_sma[0, 1] + 10, :],0.3)
            a2, b2 = get_roi(flow3[roi3_sma[1, 0] - 10:roi3_sma[1, 0] + 10, roi3_sma[1, 1] - 10:roi3_sma[1, 1] + 10, :],0.3)
            a3, b3 = get_roi(flow3[roi3_sma[2, 0] - 10:roi3_sma[2, 0] + 10, roi3_sma[2, 1] - 10:roi3_sma[2, 1] + 10, :],0.3)
            flow3_total1.append([a1 - x1, b1 - y1])
            flow3_total2.append([a2 - x1, b2 - y1])
            flow3_total3.append([a3 - x1, b3 - y1])
            flow3_total.append([a - x1, b - y1])


            gray_roi4 = gray[roi4_low - 15 : roi4_high + 10, max(0, roi4_left - 20) :roi4_right + 20]
            flow4 = cv2.calcOpticalFlowFarneback(prevgray_roi4, gray_roi4, None, 0.5, 3, 15, 5, 7, 1.5, 0)
            a, b = get_roi(flow4[bia:-bia, bia:-bia, :], 0.3)
            a1, b1 = get_roi(flow4[roi4_sma[0, 0] - 10:roi4_sma[0, 0] + 10, roi4_sma[0, 1] - 10:roi4_sma[0, 1] + 20, :],0.2)
            a2, b2 = get_roi(flow4[roi4_sma[1, 0] - 10:roi4_sma[1, 0] + 10, roi4_sma[1, 1] - 20:roi4_sma[1, 1] + 10, :],0.2)
            a3, b3 = get_roi(flow4[roi4_sma[2, 0] - 10:roi4_sma[2, 0] + 10, roi4_sma[2, 1] - 10:roi4_sma[2, 1] + 10, :],0.2)
            a4, b4 = get_roi(flow4[roi4_sma[3, 0] - 10:roi4_sma[3, 0] + 10, roi4_sma[3, 1] - 10:roi4_sma[3, 1] + 10, :],0.2)
            a5, b5 = get_roi(flow4[roi4_sma[4, 0] - 10:roi4_sma[4, 0] + 10, roi4_sma[4, 1] - 10:roi4_sma[4, 1] + 10, :],0.2)
            flow4_total1.append([a1 - x1, b1 - y1])
            flow4_total2.append([a2 - x1, b2 - y1])
            flow4_total3.append([a3 - x1, b3 - y1])
            flow4_total4.append([a4 - x1, b4 - y1])
            flow4_total5.append([a5 - x1, b5 - y1])
            flow4_total.append([a - x1, b - y1])


            gray_roi5 = gray[roi5_low-20 : roi5_high+5, max(0, roi5_left-30) : roi5_right+30]
            flow5 = cv2.calcOpticalFlowFarneback(prevgray_roi5, gray_roi5, None, 0.5, 3, 15, 5, 7, 1.5, 0)
            roi5_sma = np.array(roi5_sma)
            a1, b1 = get_roi(
                flow5[roi5_sma[0, 0]-bia*2 : roi5_sma[0, 0]+bia//2, roi5_sma[0, 1]-bia*2 : roi5_sma[0, 1]+bia, :],
                0.2)
            a2, b2 = get_roi(
                flow5[roi5_sma[1, 0]-bia*2 : roi5_sma[1, 0]+bia//2, roi5_sma[1, 1]-bia : roi5_sma[1, 1]+bia*2, :],
                0.2)
            flow5_total1.append([a1 - x1, b1 - y1])
            flow5_total2.append([a2 - x1, b2 - y1])


            gray_roi6 = gray[roi6_low:roi6_high , roi6_left:roi6_right ]
            gray_roi7 = gray[roi7_low :roi7_high , roi7_left :roi7_right ]
            if len(gray_roi6) == 0:
                a1, b1 = 0,0
            else:
                flow6 = cv2.calcOpticalFlowFarneback(prevgray_roi6, gray_roi6, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                a1, b1 = get_roi(flow6[5:-5, 5:-5, :], 0.3)
            if len(gray_roi7) == 0:
                a2, b2 = 0,0
            else:
                flow7 = cv2.calcOpticalFlowFarneback(prevgray_roi7, gray_roi7, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                a2, b2 = get_roi(flow7[5:-5, 5:-5, :], 0.3)
            flow6_total.append([a1 - x1, b1 - y1])
            flow7_total.append([a2 - x1, b2 - y1])

    flow = np.array(flow1_total).transpose(1,0)
    flow = np.concatenate((flow,np.array(flow1_total1).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow1_total2).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow1_total3).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow3_total).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow3_total1).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow3_total2).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow3_total3).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total1).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total2).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total3).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total4).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow4_total5).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow5_total1).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow5_total2).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow6_total).transpose(1,0)),axis=0)
    flow = np.concatenate((flow,np.array(flow7_total).transpose(1,0)),axis=0)

    return flow