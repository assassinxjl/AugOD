import os,time, shutil
from flask import request, jsonify
from time import sleep
from flask.views import MethodView
from concurrent.futures import ThreadPoolExecutor
from app.controllers.handlePic import PicController
from datetime import datetime
from app.controllers.zip import zipFile
from app.controllers.getFileSize import Getfile, count_files
from app.controllers.uploadToOSS import HandleOSS
from app.controllers.operateSQL import SQLController

executor = ThreadPoolExecutor()

from flask import current_app
def context_wrap(fn):
    app_context = current_app.app_context()
    def wrapper(*args, **kwargs):
        with app_context:
            return fn(*args, **kwargs)
    return wrapper


class HandlePic(MethodView):
    def post(self):
        data = request.form
        filename=data['filename']
        methodList=data['methodList']
        insertAtr=data['insertAtr']
        removeAtr=data['removeAtr']
        replaceAtr=data['replaceAtr']
        taskname = data['taskname'].strip()



        update_time = str(datetime.now()).split('.')[0]

        address, category = SQLController.find_file(filename)

        SQLController.add_info(filename, taskname, category, update_time)

        executor.submit(context_wrap(self.handle), filename, taskname, address, methodList, insertAtr, removeAtr, replaceAtr)

        result = {"code": 0,
                  "message": "",
                  "data": ""}
        result["code"] = 20000
        result["message"] = "success!"

        return jsonify(result)



    def handle(self, filename, taskname, address, methodList, insertAtr, removeAtr, replaceAtr):

        try:
            PicController.obj_aug(filename, address, methodList, insertAtr, removeAtr, replaceAtr)

            ext_dir = os.path.join(os.getcwd(), 'extend')
            ext_path = os.path.join(ext_dir, filename)
            zip_path = os.path.join(ext_dir, taskname)+'.zip'

            zipFile(ext_path, zip_path)
            filesize = Getfile(zip_path)


            h = HandleOSS()
            h.update_file(zip_path, "zip")
            h.update_files(ext_path, taskname)

            shutil.rmtree(ext_path)

            completion_time = str(datetime.now()).split('.')[0]

            print(filesize, completion_time)
            SQLController.add_state(taskname, filesize, completion_time)

            print("Task has done.")


        except Exception as ex:

            msg = "you cuo wu, shi:%s" % ex
            print(ex)




