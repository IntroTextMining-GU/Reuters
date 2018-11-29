# -*- coding: utf-8 -*-
'''
Created on Tue Nov 27 19:32:36 2018

@author: Geoff
'''
# To Activate Python 2.7 at Conda Prompt             conda activate py27
# To Deactivate Python 2.7 at Conda Prompt           conda deactivate 

import unirest
from bs4 import BeautifulSoup
import html5lib
import requests
import unicodedata

# Search terms

terms = [['castor+oil', 'castor-oil'], 
         ['castor+oil+report', 'castor-oil'],
         ['linseed+oil', 'lin-oil'],
         ['rye+prices', 'rye'],
         ['rye+closing', 'rye'],
         ['groundnut+oil', 'groundnut-oil'],
         ['sunflower+meal', 'sun-meal'], 
         ['cotton+oil', 'cotton-oil'],
         ['netherlands+guilder+florin', 'dfl'],
         ['netherlands+guilder+florin+policy', 'dfl'],
         ['copra-cake', 'copra-cake'],
         ['palm+kernel', 'palmkernel'],
         ['palm+kernel+targets', 'palmkernel'],
         ['palladium+mines', 'palladium'],
         ['palladium', 'palladium'],
         ['Norway+Krona', 'nkr'],
         ['Norway+Krona+policy', 'nkr'],
         ['South+Africa+Rand', 'rand'],
         ['South+Africa+Rand+policy', 'rand'],
         ['New+Zealand+dollar', 'nzdlr'],
         ['New+Zealand+dollar+policy', 'nzdlr'],
         ['capacity+utilization+report', 'cpu'],
         ['capacity+utilization+problems', 'cpu'],
         ['jet+fuel', 'jet'],
         ['kerosene+fuel', 'jet'],
         ['naptha', 'naptha'],
         ['naptha+targets', 'naptha'],
         ['installment+debt+problems', 'instal-debt'],
         ['consumer+credit+reports', 'instal-debt'],
         ['potato+commodity+report', 'potato'],
         ['potato+report', 'potato'],
         ['coconut', 'coconut'],
         ['coconut+report', 'coconut'],
         ['propane', 'propane'],
         ['propane+report', 'propane'],
         ['coconut-oil', 'coconut-oil'],
         ['coconut-oil+report', 'coconut-oil'],
         ['sunflower+oil', 'sun-oil'],
         ['sunflower+oil+report', 'sun-oil'],
         ['live+cattle', 'l-cattle'],
         ['live+cattle+report', 'l-cattle'],
         ['rapeseed+oil', 'rape-oil'],
         ['rapeseed+oil+report', 'rape-oil'],
         ['nickel', 'nickel'],['nickel+production+report', 'nickel'],
         ['groundnut', 'groundnut'],
         ['groundnut+report', 'groundnut'],
         ['platinum', 'platinum'],
         ['platinum+report', 'platinum'],
         ['tea', 'tea'],
         ['tea+report', 'tea'],
         ['DeutcheMark+report', 'dmk'],
         ['DeutcheMark+policy+change', 'dmk'],
         ['oat', 'oat'],
         ['oat+report', 'oat'],
         ['leading+economic+indicators+report', 'lei'],
         ['leading+economic+indicators+policy+change', 'lei'],
         ['sunflower+seed', 'sunseed'],
         ['sunflower+seed+report', 'sunseed'],
         ['lumber', 'lumber'],
         ['lumber+report', 'lumber'],
         ['income+report', 'income'],
         ['income+policy+change', 'income'],
         ['heat+production', 'heat'],
         ['heat+production+report', 'heat'],
         ['housing+report', 'housing'],
         ['housing+policy+change', 'housing'],
         ['hog', 'hog'],
         ['hog+report', 'hog'],
         ['fuel', 'fuel'],
         ['fuel+report', 'fuel'],
         ['retail+report', 'retail'],
         ['retail+policy+change', 'retail'],
         ['soy-oil', 'soy-oil'],
         ['soy-oil+report', 'soy-oil'],
         ['soy-meal', 'soy-meal'],
         ['soy-meal+report', 'soy-meal'],
         ['rapeseed', 'rapeseed'],
         ['rapeseed+report', 'rapeseed'],
         ['strategic-metal', 'strategic-metal'],
         ['strategic-metal+report', 'strategic-metal'],
         ['orange', 'orange'],
         ['orange+report', 'orange'],
         ['wholesale+price+index+report', 'wpi'],
         ['wholesale+price+index+policy+change', 'wpi'],
         ['silver', 'silver'],
         ['silver+report', 'silver'],
         ['lead', 'lead'],
         ['lead+report', 'lead'],
         ['tin', 'tin'],
         ['tin+report', 'tin'],
         ]

# Setup

articleList = []
categoryList = []

for search in range(len(terms)):

    # Contextual websearch web API building
    query = 'https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/WebSearchAPI?q=' + terms[search][0] + '&count=10&autocorrect=false'
    
    # Collecting response from website
    response = unirest.get(query, headers = {'X-RapidAPI-Key': '9a2ca87351msha85b28273336997p1619bfjsn535c8e9a3398'})
       
    # If you want to keep track of and exclude error codes
    # response.code
    
    # Accessing the list with website info
    list = response.body.get('value')
        
    for website in range(len(list)):
        newurl = list[website].get('url')
        
        # Getting new article
        try: 
            siteHTML = requests.get(newurl, timeout = 21.5)
        
            soup = BeautifulSoup(siteHTML.text, 'html.parser')
            
            # Extracting all tags for the paragraphs
            paragraphs = soup.find_all('p')
            headings = soup.find_all('h1')
            
            article = ""
            
            for h in range(len(headings)):
                article = article + unicodedata.normalize('NFKD', headings[h].text.strip()).encode('ascii','ignore') + "\n"
            
            for t in range(len(paragraphs)):
                article = article + unicodedata.normalize('NFKD', paragraphs[t].text.strip()).encode('ascii','ignore') + " "
        
            articleList.append(article)
            categoryList.append(terms[search][1])
        
        except:
            print("Timed out")
            
df = pandas.DataFrame({'articles': articleList, "category": categoryList})
df.to_csv("Extras_Articles.csv")
            

