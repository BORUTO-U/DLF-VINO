#! /usr/bin/python3
import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog
import numpy as np
import os
import cv2
import yuvplayer
import camera
import time

window = tk.Tk()
window.title("IMCL")

#屏幕居中显示
def center_window(w, h):
    # 获取屏幕 宽、高
    ws = window.winfo_screenwidth()
    hs = window.winfo_screenheight()
    # 计算 x, y 位置
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    window.geometry('%dx%d+%d+%d' % (w, h, x, y))

center_window(550,650)

workspace = "/opt/intel/2019_r1/openvino/deployment_tools/inference_engine/" \
            "samples/my_build1/intel64/workspace/workspace/workfile.cfg"

video_path = tk.StringVar()   #视频路径
ir_path = tk.StringVar()      #ir路径
rb_d = tk.StringVar()         #设备
rb_i = tk.StringVar()         #输入类型 video cam
frame_num = tk.StringVar()    #编码帧数
cfgfile = tk.StringVar()      #配置文件
QP = 27                       #编码QP值
video_width = 0               #视频宽
video_height = 0              #视频高

#输出配置文件
def write_file(filename):
    fid = open(filename, "w")
    fid.write(ir_path.get() + "\n")
    fid.write(rb_d.get() + "\n")
    fid.close()

#开始按键动作捕获
def do_inference_button():
    global video_width
    global video_height
    if video_path.get() and ir_path.get() and rb_d.get() and rb_i.get() == "VIDEO":
        os.system("clear")
        print("视频路径: %s\nIR路径: %s\n设备类型: %s\n输入类型: %s" %
               (video_path.get(), ir_path.get(), rb_d.get(), rb_i.get()))
        if text_framenum.get():
            frame_num.set(text_framenum.get())
        else:
            frame_num.set("5")
        tk.messagebox.showinfo(title='提示信息', message='默认编码5帧，开始处理...')
        write_file(workspace)
        pid = os.fork()
        if pid==0:
            time_start = time.time()
            print("开始编码...")
            os.system("/opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/my_build1/"
                  "intel64/Release/TAppEncoder -c %s -c /opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/"
                  "my_build1/intel64/workspace/encoder_intra_main.cfg -i %s -f %d -wdt %d -hgt %d -q %d"
                  % (cfgfile.get(),video_path.get(),int(frame_num.get()),video_width,video_height,QP))
            time_end = time.time()
            print("编码耗时: ",time_end - time_start)
            # window.quit()
            exit()
        else:
            return

    elif ir_path.get() and rb_d.get() and rb_i.get() == "CAM":
        os.system("clear")
        video_path.set("摄像头输入-camera_640x480.yuv")
        print("视频输入：%s\nIR路径: %s\n设备类型: %s\n输入类型: %s" %
              (video_path.get(),ir_path.get(),rb_d.get(),rb_i.get()))
        video_width = 640
        video_height = 480
        if text_framenum.get():
            frame_num.set(text_framenum.get())
        else:
            frame_num.set("5")
        write_file(workspace)
        pid = os.fork()
        if pid == 0:
            print("开始录像...")
            camera.cam_2_yuv()
            exit()

        tk.messagebox.showinfo(title='提示信息', message='默认编码5帧，开始处理...')
        pid = os.fork()
        if pid == 0:
            time_start = time.time()
            print("开始编码...")
            os.system("/opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/my_build1/"
                      "intel64/Release/TAppEncoder -c %s -c /opt/intel/2019_r1/openvino/deployment_tools/inference_engine/samples/"
                      "my_build1/intel64/workspace/encoder_intra_main.cfg -i /opt/intel/2019_r1/openvino/"
                      "deployment_tools/inference_engine/samples/my_build1/intel64/workspace/imcl_GUI/"
                      "camera_640x480.yuv -f %d -wdt %d -hgt %d -q %d" % (cfgfile.get(),int(frame_num.get()),video_width,
                                                                          video_height,QP))
            time_end = time.time()
            print("编码耗时: ", time_end - time_start)
            exit()
            #window.quit()
        else:
            return
    else:
        tkinter.messagebox.showerror(title='提示信息', message='请完善您的选择！')

#选择文件路径
def select_videopath():
    global video_width
    global video_height

    path = tk.filedialog.askopenfilename()
    video_path.set(path)

    if video_path.get():
        video_width = int(video_path.get()[-11:-8])  # 视频宽
        video_height = int(video_path.get()[-7:-4])  # 视频高

#选择ir路径
def select_irpath():
    global QP

    path = tk.filedialog.askopenfilename()
    ir_path.set(path)

    if ir_path.get():
        QP = int(ir_path.get()[-6:-4])    #QP值

#播放视频
def play_video():
    if video_width and video_height and QP:
        pid = os.fork()
        if pid == 0:
            print("播放视频...")
            yuvplayer.yuvplay('./rec.yuv',video_width,video_height,QP)
            exit()
        else:
            return
    else:
        tkinter.messagebox.showerror(title='提示信息', message='请先进行编码...')

#抽样图片
def sampling():
    if video_width and video_height and QP:
        pid = os.fork()
        if pid == 0:
            print("查看滤波效果...")
            # 临时图片文件名
            temp_1_img = "temp_1.png"
            temp_2_img = "temp_2.png"
            #保存为RGB格式
            yuvplayer.yuv_img_play('./xunfilter.yuv', video_width,video_height, temp_1_img)
            yuvplayer.yuv_img_play('./filtered.yuv', video_width,video_height,temp_2_img)

            #读取图片
            img1 = cv2.imread(temp_1_img, 1)
            img2 = cv2.imread(temp_2_img, 1)

            #添加字幕
            cv2.putText(img1, "Before filtering...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img2, "After filtering...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            #包装图片
            imgs = np.hstack([img1,img2])

            #调整窗口
            # cv2.namedWindow("滤波效果", 0)
            # cv2.resizeWindow("滤波效果", 1000, 500)
            cv2.namedWindow("滤波效果(QP=%d)" % QP, 0)
            cv2.resizeWindow("滤波效果(QP=%d)" % QP, 1000, 500)

            #展示图片
            #cv2.imshow("滤波效果", imgs)
            cv2.imshow("滤波效果(QP=%d)" % QP, imgs)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            exit()
        else:
            return
    else:
        tkinter.messagebox.showerror(title='提示信息', message='请先进行编码...')

# 设备选择按钮
# def d_radiobutton():
#     pass
#
# 输入格式选择按钮
# def i_radiobutton():
#     pass

#背景图片
photo = tk.PhotoImage(file="./fzu.gif")

#label组件
label_title=tk.Label(window, text ="基于深度学习的视频编码环路滤波模型设计", font = ("Arial", 15))
label_title.grid(row = 0, column = 1,pady=15)
label_videopath=tk.Label(window,text="视频路径：",font=("Arial",12))
label_videopath.grid(row = 4,column = 0, sticky = tk.E, pady=15)
label_irpath=tk.Label(window,text="IR路径：",font=("Arial",12))
label_irpath.grid(row = 5,column = 0, sticky = tk.E)
label_bg = tk.Label(window,image = photo,compound = tk.CENTER)
label_bg.place(relx = 0.25,rely = 0.5)
label_framenum = tk.Label(window, text = "编码帧数：",font = ("Arial",12))
label_framenum.place(relx = 0.005,rely = 0.25)

#text文本框
text_videopath = tk.Entry(window, textvariable = video_path, bg ="red", width = 55)
text_videopath.grid(row = 4,column = 1,pady=15)
text_irpath = tk.Entry(window, textvariable = ir_path, bg ="red", width = 55)
text_irpath.grid(row = 5,column = 1)
text_framenum = tk.Entry(window,textvariable = frame_num,show = None,bg = "red",width = 7)
text_framenum.place(relx = 0.15,rely = 0.25)


#button组件
b_start = tk.Button(window, text ="Do Inference", width = 15, height = 3, bg ="blue", fg ="red", command = do_inference_button)
b_start.place(relx = 0.4, rely = 0.38)

b_select_videopath = tk.Button(window,text = "...",width = 1,height = 1,bg="white",command =select_videopath)
b_select_videopath.grid(row = 4,column = 2,sticky = tk.W,pady=15)

b_select_irpath = tk.Button(window,text = "...",width = 1,height = 1,bg="white",command =select_irpath)
b_select_irpath.grid(row = 5,column = 2,sticky = tk.W)

b_sampling = tk.Button(window, text = "Sampling", width = 7, height = 1,bg = "green",command = sampling)
b_sampling.place(relx = 0.17,rely = 0.4)

b_play_video = tk.Button(window,text = "play video", width = 7, height = 1, bg = "green", command = play_video)
b_play_video.place(relx = 0.73, rely = 0.4)

#Radiobutton组件
rb_platform1 = tk.Radiobutton(window,text="OpenVINO_2019_R1",variable = cfgfile,value="/opt/intel/2019_r1/openvino/deployment_tools/"
                      "inference_engine/samples/my_build1/intel64/workspace/imcl_openvino.cfg")
rb_platform1.place(relx=0.5,rely =0.25)
rb_platform2 = tk.Radiobutton(window,text="HM16.7",variable = cfgfile,value="/opt/intel/2019_r1/openvino/deployment_tools/"
                      "inference_engine/samples/my_build1/intel64/workspace/imcl_HM.cfg")
rb_platform2.place(relx=0.80,rely =0.25)


rb_device1 = tk.Radiobutton(window, text="cpu", variable= rb_d, value ="CPU")
#rb_device1.grid(row = 6,column = 0,sticky=tk.W,pady = 20)
rb_device1.place(relx=0.01,rely=0.31)
rb_device2 = tk.Radiobutton(window, text ="vpu", variable= rb_d, value ="VPU")
#rb_device2.grid(row = 6,column = 1,ipadx=0,padx=0,sticky=tk.W)
rb_device2.place(relx=0.1,rely=0.31)
rb_device3 = tk.Radiobutton(window, text ="fpga&cpu", variable= rb_d, value ="HETERO:FPGA,CPU")
#rb_device2.grid(row = 6,column = 1,ipadx=0,padx=0,sticky=tk.W)
rb_device3.place(relx=0.19,rely=0.31)


rb_input1 = tk.Radiobutton(window, text = "video",variable = rb_i,value ="VIDEO")
#rb_input1.grid(row = 6,column = 2,sticky=tk.W)
rb_input1.place(relx=0.65,rely=0.31)
rb_input2 = tk.Radiobutton(window, text = "camera",variable = rb_i,value ="CAM")
#rb_input2.grid(row = 6,column = 3,sticky =tk.W)
rb_input2.place(relx=0.80,rely=0.31)

#canvas组件
# canvas = tk.Canvas(window,height = 200,width = 200)
# image_file = tk.PhotoImage(file = "fzu.gif")
# image = canvas.create_image(0,0,anchor = "center",image = image_file)
# canvas.place(relx = 0.35,rely = 0.4)

window.mainloop()
