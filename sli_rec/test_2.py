# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:14:19 2023

@author: PC Market
"""

import csv

array = []
array_break = []
with open('C://Users//PC Market//Documents//GitHub//fair2023//sli_rec//train_data', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        array.append(row)
        
for i in range(1, len(array)):
    if(array[i][0].split("\t")[1]!=array[i-1][0].split("\t")[1]):
        array_break.append(i)

with open('train_padding_location.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(array_break)