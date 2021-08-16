from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
 
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (MessageEvent, TextSendMessage,ImageSendMessage,TemplateSendMessage,ButtonsTemplate,MessageTemplateAction,PostbackEvent,PostbackTemplateAction)

from .scraper import stock
from datetime import date
 
line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)
 
 
@csrf_exempt
def callback(request):
 
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')
 
        try:
            events = parser.parse(body, signature)  # 傳入的事件
            print(events)
            
        except InvalidSignatureError:
            return HttpResponseForbidden()
        
        except LineBotApiError:
            return HttpResponseBadRequest()
 
        for event in events:
        
            if isinstance(event, MessageEvent):  # 如果有訊息事件
                
                if '~' not in event.message.text : 
                    
                    global stock_search
                    stock_search = stock(event.message.text) ## 導入yahoo finance api
                    stock_search.stock_name()


                    if stock_search.stock_name == 'AttributeError' : 

                         line_bot_api.reply_message(
                            event.reply_token,
                            TextSendMessage(text = '錯誤股票代號，請重新輸入')
                         )

                    else : 

                        line_bot_api.reply_message(
                            event.reply_token,
                            TextSendMessage(text = '輸入查詢期間（格式：1900-01-01~1900-01-30）')
                         )
                    
                else : 
                    ##擷取二次event內容
                    start_time = event.message.text.split('~')[0]
                    end_time = event.message.text.split('~')[1]
                    
                    try : 
                        date.fromisoformat(str(start_time))
                        date.fromisoformat(str(end_time))
                    
                        stock_search.stock_pic(start_time, end_time)

                        line_bot_api.reply_message(
                            event.reply_token,
                            [TextSendMessage(
                                text = str(stock_search.stock_id) + stock_search.stock_name + '走勢圖'),
                             ImageSendMessage(original_content_url = stock_search.pic_link,
                                         preview_image_url = stock_search.pic_link)
                            ]## 文字與圖片回應
                        )
                    
                    except :
                        
                        line_bot_api.reply_message(
                            event.reply_token,
                            TextSendMessage(text = '日期格式或值錯誤，請重新輸入')
                         )
                        
  
        return HttpResponse()
    else:
        return HttpResponseBadRequest()
    
    
    
    
    
    
    
'''    
                   
'''