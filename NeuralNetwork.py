import numpy as np
import matplotlib.pyplot as plt
from math import e
class BackPropogate:
    def __init__(self,gate):
        self.gate=gate
        self.lr=0.1
        self.epochs=1000
    def setLearningRate(self,lr):
        self.lr=lr
    def selectEpochs(self,epochs):
        self.epochs=epochs
    def sigmoid(self,x):
        return (1/(1+pow(e,-1*x)))
    def derivative(self,x):
        return (x)*(1-x)
    def train(self):
        inp=np.array([[0,0],[0,1],[1,0],[1,1]])
        if(self.gate=="and"):
            eout=np.array([[0],[0],[0],[1]])
        elif(self.gate=="or"):
            eout=np.array([[0],[1],[1],[1]])
        elif(self.gate=="xor"):
            eout=np.array([[0],[1],[1],[0]])
        hw=np.random.uniform(-1,1,(2,2))
        hb=np.random.uniform(0,1,(1,2))
        ow=np.random.uniform(-1,1,(2,1))
        ob=np.random.uniform(0,1,(1,1))
        lr=self.lr
        kk=np.array([])
        aa=np.array([])
        bb=np.array([])
        cc=np.array([])
        dd=np.array([])
        ex=0
        for i in range(self.epochs):
            hact=np.dot(inp,hw)
            hact+=hb
            hout=self.sigmoid(hact)
            oact=np.dot(hout,ow)
            oact+=ob
            oout=self.sigmoid(oact)
            err=eout-oout
            delk=err*self.derivative(oout)
            errh=err.dot(ow.T)
            delh=errh*self.derivative(hout)
            ow=ow+(hout.T).dot(delk)*lr
            hw=hw+(inp.T).dot(delh)*lr
            ob+=np.sum(delk,axis=0,keepdims=True)*lr
            hb+=np.sum(delh,axis=0,keepdims=True)*lr
            ex=100*np.sum(abs(eout-oout))/4
            kk=np.append(kk,ex)
            aa=np.append(aa,ow[0])
            bb=np.append(bb,ow[1])
            cc=np.append(cc,ob[0])
        self.ex=ex
        self.kk=kk
        self.ow1=aa
        self.ow2=bb
        self.ob=cc
        self.oo=(oout)
    def showerror(self):
        print("Error percentage is:- ",self.ex)
    def plot(self):
        plt.plot(self.kk)
        plt.xlabel("Epochs")
        plt.ylabel("Error%")
        plt.show()
    def print(self,param):
        '''ow1-Output Weights1
           ow2-Output Weights2
           ob-Output Biases
           oo-Our Output'''
        try:
            if(param=="ow1"):
                print("Output Weight 1 Array:-",self.ow1)
            elif(param=="ow2"):
                print("Output Weight 2 Array:-",self.ow2)
            elif(param=="ob"):
                print("Output Bias Array:-",self.ob)
            elif(param=="oo"):
                print("Our Output:- ",*(self.oo))
        except:
            print("Try giving valid parameters")









