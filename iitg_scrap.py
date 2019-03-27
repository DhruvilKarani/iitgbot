# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:26:53 2019

@author: DHRUVIL
"""

import requests
import bs4

def writedata(url,filename,level = None):
    sub_pages = []
    master_url = url
    freshers_website = {}
    raw  = requests.get(master_url)
    data = raw.text
    soup=bs4.BeautifulSoup(data, 'html5lib')
    if level:
        for link in soup.findAll('a'):
            if 'http' not in link['href']:
                sub_pages.append(link['href'])
    
        for page in sub_pages:
            ''' site data stored in a dictionary freshers_website
            keys = pages, value = [h1, h2, p]
            '''
            url = master_url + page
            raw  = requests.get(url)
            data = raw.text
            soup=bs4.BeautifulSoup(data, 'html5lib')
            headings_one = [x.text for x in soup.findAll('h1')]
            headings_two = [x.text for x in soup.findAll('h2')]
            headings_three = [x.text for x in soup.findAll('h3')]
            content = ' '.join([x.text for x in soup.findAll('p')])
            if not headings_one: headings_one = '<ABS>'
            if not headings_two: headings_two = '<ABS>'
            if not headings_three: headings_three = '<ABS>'
            if not content: content = '<ABS>'
            
            data = open("C:/Users/DHRUVIL/Desktop/Website/data/"+ filename+'_'+page[3:] +'.txt',"w", encoding='utf-8')
            if headings_one: data.write('<H1> '+' '.join(headings_one)+' <H1>')
            if headings_two: data.write('<H2> '+' '.join(headings_two)+' <H2>')
            if headings_three: data.write('<H3> '+' '.join(headings_three)+' <H3>')
            if content: data.write('<C> '+content+' <C>')
            data.close()
            return soup
    
#writedata('https://en.wikipedia.org/wiki/Indian_Institute_of_Technology_Guwahati','iitg_wiki.txt')
soup=writedata('http://iitg.ac.in/freshers/','sgc',level=1)

    



  