import numpy as np
import pandas as pd
import yfinance as yf
from imgurpython import ImgurClient
import pyimgur
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import mplfinance as mpf
from datetime import datetime


sns.set_style('whitegrid',{'font.sans-serif':['Arial Unicode MS','Arial']})
s = mpf.make_mpf_style(base_mpf_style='yahoo', rc = {'font.sans-serif':['Arial Unicode MS','Arial']})

class stock : 
    
    def __init__(self, stock_id) : 
        
        self.stock_id = stock_id

        
    def stock_name(self, path ='https://tw.stock.yahoo.com/q/bc?s=') : 
        
        try : 
            r = requests.get(path+ str(self.stock_id))
            soup = BeautifulSoup(r.text,"html.parser")
            self.stock_name = soup.find('tbody').find('td').find('td').find('div', {'style' : 'display: inline-block' }).text
                
        except Exception as e:
            self.stock_name = e.__class__.__name__
        
    def stock_pic(self, start_time, end_time) : 
        
        save_path = '/Users/tommy84729/Coding/Line/pic/'+ datetime.now().strftime("%m-%d-%Y, %H:%M:%S") + '.png'
        df = web.DataReader(str(self.stock_id) + '.tw', 
                            data_source = 'yahoo',
                            start = str(start_time), 
                            end = str(end_time))
        
        mpf.plot(df, type='candle', 
                 mav = (5,20,60), 
                 volume = True, 
                 style = s,
                 title = str(self.stock_id) + self.stock_name,
                 scale_padding = 0.7,
                 savefig = {'fname' : save_path,
                            'dpi' : 200})

        ## imgur
        CLIENT_ID = '41c87279269eafd'
        title = "Uploaded with PyImgur"
        im = pyimgur.Imgur(CLIENT_ID)
        uploaded_image = im.upload_image(save_path , title=title)

        self.pic_link = uploaded_image.link
        
        