# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:46:32 2019

@author: DHRUVIL
"""


import requests
import bs4
url='http://iitguwahatihelper.blogspot.com/'
raw=requests.get(url)
data=raw.text
soup=bs4.BeautifulSoup(data, 'html5lib')
