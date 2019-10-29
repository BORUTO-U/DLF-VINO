import cv2
import os
import numpy as np


def yuvplay(file_path, width, height, QP):
    np.set_printoptions(suppress=True)
    #统计帧数
    frame = 0

    #临时图片文件名
    temporary_img = "temporary.png"

    #psnr和码率
    en_info = open("/opt/intel/2019_r1/openvino/deployment_tools/"#1
                "inference_engine/samples/my_build1/intel64/workspace/workspace/en_info.txt","r")#1
    line = en_info.readline()#1
    data_list = []#1
    while line:#1
        num = list(map(float, line.split()))#1
        data_list.append(num)#1
        line = en_info.readline()#1
    en_info.close()#1
    data_array = np.array(data_list)#1

    #读取yuv序列
    fid = open(file_path, 'rb')#1
    Y = np.zeros([height, width],'UInt8')
    U = np.zeros([height >> 1, width >> 1],'UInt8')
    V = np.zeros([height >> 1, width >> 1],'UInt8')

    while True:
        if fid.readinto(Y):
            fid.readinto(U)
            fid.readinto(V)
            L=(Y).astype('float')
            Cb = (cv2.resize(U, (width, height))).reshape(height, width).astype('float')
            Cr = (cv2.resize(V, (width, height))).reshape(height, width).astype('float')
            # print(U)
            # print(cv2.resize(U, (width, height)))
            # print((cv2.resize(U, (width, height))).reshape(height, width))
            R = L + 1.402*(Cr - 128)
            G = L - 0.34414*(Cb - 128) - 0.71414*(Cr - 128)
            B = L + 1.772*(Cb - 128)
            src = np.array([B, G, R]).transpose(1,2,0).clip(0,255).astype('UInt8')
            cv2.imwrite(temporary_img,src)
            img = cv2.imread(temporary_img, 1)
            text_bitrate = "Bit_rate: %d" % data_array[frame, 0]
            text_psnr = "Psnr(Y): %.4f, Psnr(U): %.4f, Psnr(V): %.4f" \
                       %(data_array[frame, 1],data_array[frame, 2],data_array[frame, 3])
            cv2.putText(img, text_bitrate, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0),1)
            cv2.putText(img, text_psnr, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.namedWindow("滤波后视频序列: %s  QP = %d" % (file_path, QP), 0)
            cv2.resizeWindow("滤波后视频序列: %s  QP = %d" % (file_path, QP),700,480)
            cv2.imshow("滤波后视频序列: %s  QP = %d" % (file_path, QP), img)
            frame += 1#1

            if  cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
        else:
            break

    fid.close()  # 1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def yuv_s_play(file_path, width, height):
    np.set_printoptions(suppress=True)

    #临时图片文件名
    temporary_img = "temporary.png"

    #读取yuv序列
    Y = np.zeros([height, width],'UInt8')
    U = np.zeros([height >> 1, width >> 1],'UInt8')
    V = np.zeros([height >> 1, width >> 1],'UInt8')

    while True:
        # psnr和码率
        en_info = open("/opt/intel/2019_r1/openvino/deployment_tools/"#2
                       "inference_engine/samples/my_build1/intel64/workspace/workspace/en_info_temp.txt", "r")#2
        line = en_info.readline()#2
        data_list = []#2

        num = list(map(float, line.split()))#2
        data_list.append(num)#2
        en_info.close()#2
        data_array = np.array(data_list)#2
        print(data_array)#2

        fid = open(file_path, 'rb')#2
        if fid.readinto(Y):
            fid.readinto(U)
            fid.readinto(V)
            L=(Y).astype('float')
            Cb = (cv2.resize(U, (width, height))).reshape(height, width).astype('float')
            Cr = (cv2.resize(V, (width, height))).reshape(height, width).astype('float')
            # print(U)
            # print(cv2.resize(U, (width, height)))
            # print((cv2.resize(U, (width, height))).reshape(height, width))
            R = L + 1.402*(Cr - 128)
            G = L - 0.34414*(Cb - 128) - 0.71414*(Cr - 128)
            B = L + 1.772*(Cb - 128)
            src = np.array([B, G, R]).transpose(1,2,0).clip(0,255).astype('UInt8')
            cv2.imwrite(temporary_img,src)
            img = cv2.imread(temporary_img, 1)
            text_bitrate = "Bit_rate: %d" % data_array[0, 0]
            text_psnr = "Psnr(Y): %.4f, Psnr(U): %.4f, Psnr(V): %.4f" \
                       %(data_array[0, 1],data_array[0, 2],data_array[0, 3])
            cv2.putText(img, text_bitrate, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0),1)
            cv2.putText(img, text_psnr, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.namedWindow(file_path, 0)
            cv2.resizeWindow(file_path,700,480)
            cv2.imshow(file_path, img)
            fid.close()  # 2

            if  cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
        else:
            break

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def yuv_img_play(filepath, width, height, imgname):
    #读取yuv图片
    Y = np.zeros([height, width],'UInt8')
    U = np.zeros([height >> 1, width >> 1],'UInt8')
    V = np.zeros([height >> 1, width >> 1],'UInt8')

    fid = open(filepath, "rb")
    if fid.readinto(Y):
        fid.readinto(U)
        fid.readinto(V)
        L = (Y).astype('float')
        Cb = (cv2.resize(U, (width, height))).reshape(height, width).astype('float')
        Cr = (cv2.resize(V, (width, height))).reshape(height, width).astype('float')

        R = L + 1.402 * (Cr - 128)
        G = L - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
        B = L + 1.772 * (Cb - 128)

        src = np.array([B, G, R]).transpose(1, 2, 0).clip(0, 255).astype('UInt8')
        cv2.imwrite(imgname, src)
    fid.close()




#yuvplay('/opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/'
        #'my_build1/intel64/workspace/workspace/rec.yuv',320,256)
#yuvplay('/opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/'
        #'my_build1/intel64/workspace/workspace/filtered.yuv',320,256)


