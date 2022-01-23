import time
from fastapi import FastAPI, File,UploadFile,BackgroundTasks,Request
from PIL import Image
from io import BytesIO
import imghdr
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
#import matplotlib.pyplot as plt 
import random
from starlette.middleware.cors import CORSMiddleware
from config import get_config_data

app = FastAPI()

# 通配符匹配，允许域名和方法
app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"], #origins,   
    allow_credentials=True, 
    allow_methods=["*"],   
    allow_headers=["*"],   
)

base_dir   = get_config_data()['base_dir'] #'F:\\py_experiment\\'
model_path = base_dir+'model.h5'
model      = load_model(model_path)
cates      = ['分类1','分类2','分类3','分类4']
valid_ext  = ['jpeg','jpg','png']
#------------------函数区start-----------------------------------

def write_log(log_file_dir:str,message: str):
    with open(log_file_dir, mode="a") as log:
        log.write(message)
def save_file(file_dir:str,data:bytes):
    with open(file_dir, "wb") as f:
            f.write(data)
def error_res(code:int ,data:dict):
    return {
            "code": code,
            "data":data
        }
#------------------函数区end-----------------------------------

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}



@app.post("/upload/")
async def create_file( bg_tasks: BackgroundTasks,request: Request, file: bytes = File(...) ):
    time1 = time.time()
    # ext = file.content_type
    # ext = ext.split('/')[1]
    # contents = await file.read()
    ext  = imghdr.what(None,file)
    file_len = request.headers['content-length']
    #print(file_len)
    if int(file_len) > (1024*1024):
        error_data = {
                "msg" : '文件应小于1M',
                'consume': str(round( time.time()-time1 ,3))+'s',
            } 
        return error_res(1,error_data)
    if ext not in valid_ext:
        return error_res(1,{"msg":'文件格式错误,请上传jpg、jpeg、png格式的图片','consume': str(round( time.time()-time1 ,3))+'s' })
    log_data = time.strftime("%H:%M:%S", time.localtime())+' ' +request.client.host+':'+str(request.client.port)+ ' - '
    try:
        tmp_pic = "./temp_pic/"+str(random.random())+"."+ext
        save_file(tmp_pic,file)
        # img = load_img(tmp_pic, target_size=(150, 150))
        #os.unlink(tmp_pic)

        img = Image.open(BytesIO(file)).resize((150, 150))
        #print(img.mode)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        x   = img_to_array(img)  
        x   = x.reshape((1,) + x.shape) 
        x  /= 255

        # print(img.mode)
        # print(x.shape)
        feature_maps  = model.predict(x)
        predict_index = np.argmax(feature_maps)
        rate   = str( round(max(feature_maps[0])*100,3) )+'%'
        result = cates[predict_index] 
        
        bg_tasks.add_task(write_log,'./log_file/'+time.strftime("%Y-%m-%d", time.localtime()) +'_request.txt',log_data+ str(request.scope)+'\r\n')
        return {
            "code": 0,
            "data":{
                "rate" : rate,
                "name" : result,
                'cate_id':str(predict_index),
                'consume': str(round( time.time()-time1 ,3))+'s',
                'ext':ext,
                'show_dir':tmp_pic
            } 
        }
    except Exception as e:
        bg_tasks.add_task(write_log,'./log_file/'+time.strftime("%Y-%m-%d", time.localtime()) +'_error.txt',log_data + str(e)+'\r\n')
        error_data = {
                "msg" : str(e),
                'consume': str(round( time.time()-time1 ,3))+'s',
                'ext':ext,
                'show_dir':tmp_pic
            } 
        return error_res(1,error_data)