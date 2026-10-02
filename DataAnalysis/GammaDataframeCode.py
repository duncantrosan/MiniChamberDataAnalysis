# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:51:44 2026

@author: dptro
"""

import pandas as pd
import matplotlib.pyplot as plt

HairpinNotIn = pd.read_csv("G:\My Drive\MiniChamberData\Data\HairpinProbePrelimData\HairpinNotIn.csv")
HairPinIn = pd.read_csv("G:\My Drive\MiniChamberData\Data\HairpinProbePrelimData\HairPinProbeInsideChamber.csv")

FrequencyIn  = HairPinIn['! COPPER MOUNTAIN TECHNOLOGIES'][3:].astype('float').values
FrequencyOut = HairpinNotIn['! COPPER MOUNTAIN TECHNOLOGIES'][3:].astype('float').values


S11In  = HairPinIn[' R140'][3:].astype('float').values
S11Out = HairpinNotIn[' R140'][3:].astype('float').values

plt.figure()
plt.plot(FrequencyIn*10**-9,S11In,label='Hairpin probe inside chamber (No Plasma)') 
plt.plot(FrequencyOut*10**-9,S11Out,label = 'Hairpin probe outside chamber')
plt.legend()
plt.xlabel('Frequency / GHz')
plt.ylabel('S11 / dB')