# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:55:46 2019

@author: Kuntal
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 22:41:31 2019

@author: Kuntal
"""
import pickle,re
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import MaxEntClassifier
from textblob.classifiers import DecisionTreeClassifier
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog,Canvas,PhotoImage
from PIL import ImageTk, Image
### Reading all training data 
# 1. read anger-ratings-0to1.txt
def GUI_FIRST():
    win=tk.Tk()
    win.geometry("300x250")
    win.title("Sentiment Analysis")
    w=tk.Label(win,text="DIGITEXT\n",font=("Arial Bold",30))
    w1=tk.Label(win,text="Forensic Text Validation\n\n")
    w.pack()
    w1.pack()
    photo=tk.PhotoImage(file="H:\\EmotionDetection\\GUI\\GUICode\\priya gui\\file.png")
    labelphoto=tk.Label(win,image=photo)
    labelphoto.pack()
    win.after(3000,win.destroy)
    win.mainloop()

def GUI():
    root=tk.Tk()
    root.geometry("400x300")
    topframe=tk.Frame(root)
    entry=tk.Entry(topframe)
    entry.insert(tk.END," Select Any Text File")
    entry.pack()
    #entry.grid(row=0, column=0)
     
    def open_file():
        TestFile=filedialog.askopenfilename()
        model=LoadModel("model_NB.pkl")
        Prediction(TestFile,model)
    button=tk.Button(root,text="open file",command=open_file,height = 1, width = 14)
    button.pack()
    topframe.pack()
    def file1():
        master = tk.Toplevel()
        master.geometry("200x150")
        master.title("Show Analysis")
        img=Image.open("H:\\Project_CDAC\\Target\\Emotion.png")
        img = img.resize((130,130),Image.ANTIALIAS)
        photoImg = ImageTk.PhotoImage(img)
       
        b = tk.Button(master,image=photoImg, width=120)
        b.pack()
        master.mainloop()
          
    button2=tk.Button(root,text="analize emotion",command=file1,height = 1, width = 14)
    button2.pack()    
    def file2():  
        window = tk.Tk()
        window.title("Show Report")
        window.geometry("1900x400")
        lbl = tk.Label(window)
        lbl.grid(column=0,row =0)
        read_file=open("H:\\Project_CDAC\\Target\\Report.txt",'r',encoding="UTF8")
        res = read_file.read()
        lbl.configure(text=res)
        #lbl.grid(column=0,row =20)
        lbl.pack()
        window.mainloop()
        
    button3=tk.Button(root,text="Show Report",command=file2,height = 1, width = 14).pack()
    root.mainloop()

def Train():
    train_file_anger = open("H:\\EmotionDetection\\Saif mohammad\\taining set\\anger-ratings-0to1.txt",'r',encoding="utf8",);
    train = [] 
    value_set = 20
    i = 0
    for line in train_file_anger.readlines():      
        for element in line[5:-1].split('\n'):
           # train.append((element,'anger'))
           if i==value_set:
               break
           a = element.split('anger')
           train.append((a[0].lstrip(),'anger'))
           i = i+1
    i=0
    train_file_fear =  open("H:\\EmotionDetection\\Saif mohammad\\taining set\\fear-ratings-0to1.txt",'r',encoding="utf8",);
    for line in train_file_fear.readlines():      
        for element in line[5:-1].split('\n'):
           if i==value_set:
               break
           a = element.split('fear')
           train.append((a[0].lstrip(),'fear'))
           i = i+1
           
    i=0
    train_file_joy =  open("H:\\EmotionDetection\\Saif mohammad\\taining set\\joy-ratings-0to1.txt",'r',encoding="utf8",);
    for line in train_file_joy.readlines():      
        for element in line[5:-1].split('\n'):
           if i==value_set:
               break
           a = element.split('joy')
           train.append((a[0].lstrip(),'joy'))
           i = i+1
    i=0
    train_file_sadness =  open("H:\\EmotionDetection\\Saif mohammad\\taining set\\sadness-ratings-0to1.txt",'r',encoding="utf8",);
    for line in train_file_sadness.readlines():      
        for element in line[5:-1].split('\n'):
           if i==value_set:
               break
           a = element.split('sadness')
           train.append((a[0].lstrip(),'sadness'))
           i = i+1
    value_set = 20            
    test = []
    i=0 
    test_file_anger = open("H:\\EmotionDetection\\Saif mohammad\\test set\\with intensity labels\\anger.txt",'r',encoding="utf8",);
    for line in test_file_anger.readlines():      
        for element in line[5:-1].split('\n'):
            if i ==value_set:
                break
            #test.append((element,'anger'))
            b= element.split('anger')
            test.append((b[0].lstrip(),'anger'))
            i= i+1
    i=0
    test_file_fear = open("H:\\EmotionDetection\\Saif mohammad\\test set\\without intensity labels\\fear.txt",'r',encoding="utf8",);
    for line in test_file_fear.readlines():      
        for element in line[5:-1].split('\n'):
            if i ==value_set:
                break 
            b= element.split('fear')
            test.append((b[0].lstrip(),'fear'))
            i=i+1
    i=0        
    test_file_joy = open("H:\\EmotionDetection\\Saif mohammad\\test set\\without intensity labels\\joy.txt",'r',encoding="utf8",);
    for line in test_file_joy.readlines():      
        for element in line[5:-1].split('\n'):
            if i ==value_set:
                break 
            b= element.split('joy')
            test.append((b[0].lstrip(),'joy'))
            i= i+1
    i=0        
    test_file_sadness = open("H:\\EmotionDetection\\Saif mohammad\\test set\\without intensity labels\\sadness.txt",'r',encoding="utf8",);
    for line in test_file_sadness.readlines():      
        for element in line[5:-1].split('\n'):
            if i ==value_set:
                break 
            b= element.split('sadness')
            test.append((b[0].lstrip(),'sadness'))
            i= i+1


    model =DecisionTreeClassifier(train)
    print("accuracy label of Naive Bayes Classifier:{:.4f}".format(model.accuracy(test)))
    print("Training completed....")
    #Dumping model NaiveBayes
    fp=open("H:\\Project_CDAC\\Models\\model_NB.pkl","wb")
    pickle.dump(model,fp)
    fp.close()
    print("Serialization of model completed")
    

def LoadModel(modelname):
    fp=open("H:\\Project_CDAC\\Models\\"+modelname,"rb")
    model=pickle.load(fp)
    fp.close()
    print("Model Loaded Successfully")
    return model    


def Prediction(testfilenm,model):
    count_anger =0
    count_fear =0 
    count_joy =0
    count_sadness= 0
    total_count=0
    print("start prediction method ")
    fp=open("H:\\Project_CDAC\\Target\\Report.txt","w")
   #ftp.storbinary("STOR " + i, file)
   # print("---->>>"+type(testfilenm))
    test_read=open(testfilenm,"r",encoding="utf8")
    cont=test_read.read()
    print("start split ")
    lns=re.split("[.\n\?]",cont) #Need to implement splitter for normal and social media text
    for l in lns:
        l=l.strip()
        #print (l)
        pc=model.classify(l)
        if pc== "anger":
            count_anger = count_anger+1
        elif pc == "fear":
            count_fear = count_fear+1
        elif pc == "joy":
            count_joy = count_joy+1
        elif pc == "sadness":
            count_sadness = count_sadness+1
        fp.write(l.strip()+"\t"+"->"+pc+"\n")
    fp.close()
    total_count=count_anger+count_fear+count_joy+count_sadness
    print("anger :",count_anger)
    print("fear:",count_fear)
    print("joy :",count_joy)
    print("sadness :",count_sadness)
    sentiment_plot_x = ('anger','fear','joy','sadess')
    sentiment_plot_y = ((count_anger*100.0)/total_count,(count_fear*100.0)/total_count,(count_joy*100.0)/total_count,(count_sadness*100.0)/total_count)
    print("Start plot ")
    plt.bar(sentiment_plot_x,sentiment_plot_y,align='center')
    plt.savefig('H:\\Project_CDAC\\Target\\Emotion.png', format='png', dpi=1200)
    plt.show()
    print("End of prediction ")
if __name__ == "__main__":
    #TestFile='H:\\EmotionDetection\\Saif mohammad\\Source\\test.txt'
    #Train()
    #model=LoadModel("model_NB.pkl")
    #Prediction(TestFile,model)
    GUI_FIRST()
    GUI()
    
