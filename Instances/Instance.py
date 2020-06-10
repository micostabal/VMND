from abc import ABC, abstractmethod, ABCMeta
from gurobipy import *
import os

class Instance(metaclass = ABCMeta):
 
    def __init__(self, instName = ''):
        super().__init__()

    @abstractmethod
    def createInstance(self): pass

    @abstractmethod
    def exportMPS(self, writePath = os.path.join(os.path.pardir, 'MIPLIB')): pass

    @abstractmethod
    def genNeighborhoods(self): pass

    @abstractmethod
    def genLazy(self): pass

    @abstractmethod
    def genTestFunction(self): pass

    @abstractmethod
    def run(self): pass

    @abstractmethod
    def analyzeRes(self): pass

    @abstractmethod
    def visualizeRes(self): pass
 

if __name__ == '__main__': pass
