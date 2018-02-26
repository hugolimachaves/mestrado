#!/usr/bin/env python
from tkinter import *
from PIL import ImageTk
from PIL import Image
import argparse
import os, sys
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import math
import numpy as np
import matplotlib.pyplot as plt
import binary_features as bf
import key_frame as kf

resize_factor = 3

class Viewer:
    def __init__(self, master, filelist, fc7):
        self.top = master
        self.files = filelist
        self.fc7 = fc7
        self.index = 0
        self.binary_frames = bf.codify_frames(fc7.values, 16)
        self.key_frames = []
        #display first image
        filename = filelist[0]
        if not os.path.exists(filename):
            self.top.quit()

        #fc7_array = fc7.loc[0, :]
        fc7_array = fc7.loc[0, :] - fc7.loc[len(self.files) - 1, :] 
        fc7_array = fc7_array.values.reshape([int(math.sqrt(fc7_array.shape[0])),int(math.sqrt(fc7_array.shape[0]))])

        #print(max(fc7.loc[0, :]), min(fc7.loc[0, :]))
        #norm = mpl.colors.Normalize(vmin=min(fc7.loc[0, :]), vmax=max(fc7.loc[0, :]))
        #cmap = cm.hot
        #m = cm.ScalarMappable(norm=norm, cmap=cmap)
        #arr = m.to_rgba(fc7_array)
        # Make plot with vertical (default) colorbar
        fig, ax = plt.subplots()
        cax = ax.imshow(fc7_array, interpolation='nearest', cmap=cm.coolwarm)

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax) 

        plt.savefig(os.path.basename(filename) + '_graph.png', transparent=True)
        fc7img = Image.open(os.path.basename(filename) + '_graph.png')
        width, height = fc7img.size
        fc7img = fc7img.resize((620, 440))
        self.tkfc7 = ImageTk.PhotoImage(fc7img, palette=256)
        fc7img.close()
        os.remove(os.path.basename(filename) + '_graph.png')

        self.container1 = Frame(master)
        self.container1["pady"] = 10
        self.container1.pack()
		
        # container da hash
        self.container4 = Frame(master)
        self.container4["pady"] = 10
        self.container4.pack()

        # container do campo de busca e do outro botao
        self.container5 = Frame(master)
        self.container5["pady"] = 10
        self.container5.pack()

        # container dos keyframes
        self.container6 = Frame(master)
        self.container6["pady"] = 10
        self.container6.pack()

        self.container2 = Frame(master)
        self.container2["pady"] = 10
        self.container2["padx"] = 10
        self.container2.pack()

        self.container3 = Frame(master)
        self.container3["pady"] = 10
        self.container3.pack()
		
        

        self.txtfilename= Label(self.container1, text=os.path.dirname(filename))
        self.txtfilename.pack()

        back = Button(self.container1, text="<<", command=lambda : self.nextframe(-1))
        back.pack(side='left')
        next = Button(self.container1, text=">>", command=lambda : self.nextframe(1))
        next.pack(side='left')

        # nome do arquivo do frame
        self.txtframe = StringVar()
        self.txtframe.set(os.path.basename(filename))
        self.txtframeF = Entry(self.container1, textvariable=self.txtframe)
        self.txtframeF.pack(side='left')
		
        # settando a hash incial
        self.txtbin = StringVar()
        self.txtbin.set(self.binary_frames[self.index])
        self.binary_feat = Label(self.container4, textvariable=self.txtbin)
        self.binary_feat.pack()

        # valor da distancia de hamming e valor minimo dos fragmentos
        self.txtHam = StringVar()
        self.txtHam.set('1 20')
        self.txtHamF = Entry(self.container5, textvariable=self.txtHam)
        self.txtHamF.pack(side='left')

        # bostao para recalcular os keyframes
        self.btnKey = Button(self.container5, text="Key frame")
        self.btnKey["command"] = self.Key_Frame_Calc
        self.btnKey.pack(side=RIGHT)

        # settando os keyframes iniciais
        self.txtKey = StringVar()
        self.key_frames = kf.calculate_key_frames(self.binary_frames, 1, 20)
        self.txtKey.set(self.key_frames)
        self.key_frame = Label(self.container6, textvariable=self.txtKey)
        self.key_frame.pack()

        # imagem do frame
        im = Image.open(filename)
        width, height = im.size
        im = im.resize((height * resize_factor, height * resize_factor))
        self.size = im.size
        self.tkimage = ImageTk.PhotoImage(im, palette=256)
		
        # imagem do frame
        im = Image.open(filename)
        width, height = im.size
        im = im.resize((height * resize_factor, height * resize_factor))
        self.size = im.size
        self.tkimage = ImageTk.PhotoImage(im, palette=256)
        

        self.lbl = Label(self.container2, image=self.tkimage, borderwidth=1, background="black")
        self.lbl.pack(side='left')
		
        # imagem do keyframe anterior
        im2 = Image.open(self.searchPreviousKey(filename))
        width, height = im2.size
        im2 = im2.resize((height * resize_factor, height * resize_factor))
        self.size = im2.size
        self.tkimage2 = ImageTk.PhotoImage(im2, palette=256)
        

        self.lbl2 = Label(self.container2, image=self.tkimage2)
        self.lbl2.pack(side='left')

        # nome do arquivo da fc7
        self.lbl2 = Label(self.container2, image=self.tkfc7)
        self.lbl2.pack(side='right')

        self.btnBuscar = Button(self.container1, text="Buscar")
        self.btnBuscar["command"] = self.search
        self.btnBuscar.pack(side=RIGHT)


        # the button frame
        fr = Frame(master)
        fr.pack(side='bottom', expand=1, fill='x')

    def searchPreviousKey(self, filename):
        
        actual_frame = int((os.path.basename(filename)).split('.')[0])
        previous_key = self.key_frames[-1]
        for item in self.key_frames:
            if actual_frame < item:
                return os.path.join(os.path.dirname(filename), str(previous_key-1)+'.png')
            previous_key = item

    def getImage(self, filename):
        im = Image.open(filename)

    def nextframe(self,i=1, imgnum=-1):
        if imgnum == -1:
            self.index += i
        else:
            self.index = imgnum - 1
        if self.index >= len(self.files):
            self.index = 0
        elif self.index < 0:
            self.index = len(self.files) - 1
        filename = self.files[self.index]
        #if not os.path.exists(filename):
        #    self.top.quit()
        self.txtframe.set(os.path.basename(filename))
        self.txtbin.set(self.binary_frames[self.index])
        #self.evar.set(self.index+1)
        
        im = Image.open(filename)
        width, height = im.size
        im = im.resize((height * resize_factor, height * resize_factor))
        self.tkimage.paste(im)
		
        #Atualizando o ultimo key frame
        im2 = Image.open(self.searchPreviousKey(filename))
        width, height = im2.size
        im2 = im2.resize((height * resize_factor, height * resize_factor))
        self.tkimage2.paste(im2)

        #fc7_array = fc7.loc[self.index, :]
        if self.index == 0:
            fc7_array = fc7.loc[self.index, :] - fc7.loc[len(self.files) - 1]    
        else:
            fc7_array = fc7.loc[self.index, :] - fc7.loc[self.index - 1, :] 

        fc7_array = fc7_array.values.reshape([int(math.sqrt(fc7_array.shape[0])),int(math.sqrt(fc7_array.shape[0]))])
        fig, ax = plt.subplots()
        cax = ax.imshow(fc7_array, interpolation='nearest', cmap=cm.coolwarm)

        cbar = fig.colorbar(cax) 

        plt.savefig(os.path.basename(filename) + '_graph.png', transparent=True)
        plt.close()
        fc7img = Image.open(os.path.basename(filename) + '_graph.png')
        
        width, height = fc7img.size
        fc7img = fc7img.resize((620, 440))
        self.tkfc7.paste(fc7img)
        fc7img.close()
        os.remove(os.path.basename(filename) + '_graph.png')

        if(int(str(self.txtframeF.get()).split('.')[0])+1 in self.key_frames):
            self.lbl.config(background="red")
        else:
            self.lbl.config(background="black")

    def search(self):
        filename = os.path.join(os.path.dirname(self.files[0]),self.txtframeF.get())
        if os.path.exists(filename):
            im = Image.open(filename)
            width, height = im.size
            im = im.resize((height * resize_factor, height * resize_factor))
            self.index = int (os.path.splitext(self.txtframeF.get())[0])
            self.tkimage.paste(im)
			
            # Atualizando ultimo key frame
            im2 = Image.open(self.searchPreviousKey(filename))
            width, height = im2.size
            im2 = im2.resize((height * resize_factor, height * resize_factor))
            self.tkimage2.paste(im2)
            if(int(str(self.txtframeF.get()).split('.')[0])+1 in self.key_frames):
                self.lbl.config(background="red")
            else:
                self.lbl.config(background="black")
			
    def Key_Frame_Calc(self):
        h_dist, min_size = str(self.txtHam.get()).split()
        self.key_frames = kf.calculate_key_frames(self.binary_frames, int(h_dist), int(min_size))
        self.txtKey.set(self.key_frames)
		

def openImages(folder):
    if not os.path.exists(folder):
        print("Error open folder")
	
    pasta = folder
    list_folder = []
    nomes = os.listdir(pasta)
    
    caminhos = [os.path.join(pasta, str(nome) + '.png') for nome in range(0,len(nomes))]
    arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
    for arq in arquivos:
        list_folder.append(arq)
    return list_folder

def readfc7(filename):
    
    fc7 = pd.read_csv(filename, header=None)
    return fc7  
    
# --------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video", action='store', type=str, help="Video path", default='list')
    parser.add_argument("fc7", help= "Input params file", default="params_dnn")
    filelist = openImages(parser.parse_args().video)
    fc7 = readfc7(parser.parse_args().fc7)

    root = Tk()
    app = Viewer(root, filelist, fc7)
    root.mainloop()
    plt.close()

