#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:24:13 2020

@author: rohith
"""

import shutil

for c in range(40):
    if c!= 8:
        src = "Training/s" + str(c+1) + "/9.pgm"
        dst = "Testing/s" + str(c+1) + "/9.pgm"
        shutil.move(src,dst)