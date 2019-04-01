# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:05:58 2019

@author: DHRUVIL
"""

reverse=''
stringone = "keep calm and clean data how are you"
temp=''
for char in stringone:
    if char != ' ':
        temp+=char
    if char == ' ':
        char+=temp
        char+=reverse
        reverse=char
        temp=''
temp+=reverse     
print(temp)