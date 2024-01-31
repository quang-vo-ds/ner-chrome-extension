import requests
from time import sleep
import re
import json
import os
from bs4 import BeautifulSoup

class Crawler:
    def __init__(self,
                 headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36" ,'referer':'https://www.google.com/'}
                 ):
        self.headers = headers  ## define header to access google scholar website

    ## Title of the website
    def get_title(self, url):
        doc = self.get_web_info(url)
        title_tag = self.get_tags(doc)
        title = title_tag.get_text()
        return title
    
    ## Get website info
    def get_web_info(self, url):
        response = requests.get(url,headers=self.headers) # download the page
        if response.status_code != 200: # check successful response
            print('Status code:', response.status_code)
            raise Exception('Failed to fetch web page ')
        #parse using beautiful soup
        doc = BeautifulSoup(response.text,'html.parser')
        return doc
    
    ## Extracting information of the tags
    def get_tags(self, doc):
        title_tag = doc.find('title')
        return title_tag
    
if __name__ == '__main__':
    url = 'https://www.medicalnewstoday.com/articles/psoriasis-and-chronic-inflammation'
    crawler = Crawler()
    print(crawler.get_title(url))