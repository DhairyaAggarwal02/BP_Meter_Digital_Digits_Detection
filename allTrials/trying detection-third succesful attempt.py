import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import cv2
import PIL
from PIL import Image,ImageTk

window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")
window.geometry("500x500")
window.state("zoomed")  #put window in full screen

imageFrame1 = tk.Frame(window, width=200, height=300)
imageFrame1.place(anchor='nw', height='450', width='450', x='90', y='100')
imageFrame1.configure(height='200', width='200')

lmain1 = tk.Label(imageFrame1,width=200, height=300,bg="purple")
lmain1.place(anchor='nw', height='450', width='450', x='0', y='0')
lmain1.configure(font='TkTextFont', justify='left')

label1=Label(window,text="Camera",padx=202,pady=9,bg="orange")
label1.place(x="90",y="57")

imageFrame3 = tk.Frame(window, width=220, height=200)
imageFrame3.place(anchor='nw', x='570', y='100')
imageFrame3.configure(height='200', width='220')

lmain3 = tk.Label(imageFrame3, width=220, height=200,bg="purple")
lmain3.place(anchor='nw', x="0", y="0")
lmain3.configure(font='TkTextFont', justify='left', text='label3')

label2=Label(window,text="Region Of Interest",padx=60,pady=9,bg="orange")
label2.place(x="570",y="57")


def yolov4():
    def finalDetectImage():
        global frame3
        height, width, channels = frame3.shape
        font = cv2.FONT_HERSHEY_PLAIN
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    #Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    #Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    #print(detection)
                    #print(class_id)
                    
        global L1
        L1=[]
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
        for i in range(len(boxes)):              
            if i in indexes:              
                x, y, w, h = boxes[i]
                global LABEL                       
                LABEL = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)       
                cv2.putText(frame3, LABEL, (x, y + 30), font, 3, color, 3)
                #print("detections are")
                for z in LABEL:
                    L1.append(z)
                    #print(z)
        print(L1)
        global FINAL
        FINAL = ''.join(map(str, L1))
        print(FINAL)
    finalDetectImage()
    
def OpenCamera():
    cap = cv2.VideoCapture(0)
    def show_frame():
      global frame
      ret,frame = cap.read()
      frame = cv2.flip(frame, 1)
      cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      img = Image.fromarray(cv2image)
      imgtk = ImageTk.PhotoImage(image=img)
      lmain1.imgtk = imgtk
      lmain1.configure(image=imgtk)
      lmain1.after(10, show_frame)
    show_frame()

def ShowROI():
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    x,y,w,h=cv2.selectROI(frame)
    #print(x,y,w,h)
    #x,y,w,h=223,197,247,283
    def SelectROI():
        ret,frame=cap.read()
        global frame3
        frame3=frame[(y):(y+h),(x):(x+w)]
        #frame3 = cv2.flip(frame3, 1)
        cv2image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        img3 = Image.fromarray(cv2image3)
        imgtk3 = ImageTk.PhotoImage(image=img3)
        lmain3.imgtk3 = imgtk3
        lmain3.configure(image=imgtk3)
        lmain3.after(10,SelectROI)
    SelectROI()

def PRINT():
    global FINAL
    global e1
    e1=Entry(window)
    e1.place(x="820",y="100")
    e1.insert(0,FINAL)
    
def detect():
    
    # load yolo
    net = cv2.dnn.readNet("5000 iterations.weights", "yolov4-custom.cfg")
    global classes
    classes = ["0","1","2","3","4","5","6","7","8","9"] 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    global colors
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    global frame3
    blob = cv2.dnn.blobFromImage(frame3, 0.00392, (320,320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    global outs
    outs = net.forward(output_layers)
    yolov4()
    PRINT()


    
def save_info():
    response=messagebox.askyesno("Message from NPL-CSIR","Do you want to save record")
    print(response)
    if response==True:
        detection1=e1.get()
        print(detection1)
        file=open("user.txt","a")
        file.write("Detection : " + detection1 + "\n")
        messagebox.showinfo("Message","Record Saved")
    elif response==False:
        pass
    

    
btn1 = Button(window,text= "open camera", command=OpenCamera,padx=75,pady=7,relief="ridge")
btn1.place(x="90",y="560")

btn2 = Button(window,text= "detect", command=detect,padx=27,pady=7,relief="ridge")
btn2.place(x="570",y="309")

btn3 = tk.Button(window,command=ShowROI)
btn3.configure(relief="ridge",text="Select ROI",padx=75,pady=7)
btn3.place(x="330",y="560")


btn4 = Button(window,text="Save information",command=save_info,padx=7,pady=7,relief="ridge")
btn4.place(x="678",y="309")
