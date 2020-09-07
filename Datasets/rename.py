#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:29:10 2020

@author: rohith
"""

import os

for c in range(1,40):
    old_name = "Testing/s" + str(c+1) + "/" + str(c+1) + ".pgm"
    new_name = "Testing/s" + str(c+1) + "/1.pgm"
    os.rename(old_name,new_name)
    
for c in range(40):
    if c!=8:
        old_name = "Testing/s" + str(c+1) + "/9.pgm"
        new_name = "Testing/s" + str(c+1) + "/2.pgm"
        os.rename(old_name,new_name)