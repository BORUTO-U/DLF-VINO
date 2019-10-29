import numpy as np
import cv2

def cam_2_yuv():
    cap = cv2.VideoCapture(0)

    open('./camera_640x480.yuv', 'wb+').close()
    width=640
    height=480
    wait = 0
    while True:
        fid = open('./camera_640x480.yuv', 'ab')
        ret, frame = cap.read()
        wait+=1
        if wait < 7:
            continue
        if ret == 1:
            data = frame
            R = data[:,:,2]
            G = data[:,:,1]
            B = data[:,:,0]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            U = - 0.1687 * R - 0.3313 * G + 0.5 * B + 128
            V = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

            #Y = Y.numpy()
            Y = cv2.resize(Y, (width, height))
            fid.write(Y.clip(0, 255).round().astype('uint8'))
            #U = U.numpy()
            U = cv2.resize(U, (width>>1, height>>1))
            fid.write(U.clip(0, 255).round().astype('uint8'))
            #V = V.numpy()
            V = cv2.resize(V, (width>>1, height>>1))
            fid.write(V.clip(0, 255).round().astype('uint8'))

            cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("录像结束...")
                break
        fid.close()

    cap.release()
    cv2.destroyAllWindows()
