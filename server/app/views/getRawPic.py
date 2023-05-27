from flask.views import MethodView
import io
from PIL import Image
from flask import jsonify,request
from app.controllers.operateSQL import SQLController

class GetRawPic(MethodView):
    def post(self):
        param = request.args
        print(param)
        filename = param.get("filename")
        address, category = SQLController.find_file(filename)
        with open(address, 'rb') as f:
            a = f.read()
        '''对读取的图片进行处理'''
        img_stream = io.BytesIO(a)
        img = Image.open(img_stream)

        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="JPEG")
        imgByteArr = imgByteArr.getvalue()

        return imgByteArr

