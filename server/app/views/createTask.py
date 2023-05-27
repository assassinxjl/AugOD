import os,time
from flask import request, jsonify
from flask.views import MethodView
from app.controllers.createTask import TaskController
from datetime import datetime
from app.controllers.zip import zipFile
from app.controllers.getFileSize import Getfile, count_files
from app.controllers.operateSQL import SQLController

class CreateTask(MethodView):
    def post(self):
        data = request.form
        taskname=data['taskname']
        desc = data['desc']
        filename=data['filename']
        methodList=data['methodList']
        multiple=int(data['multiple'])
        timestamp=str(int(time.time()))

        update_time = str(datetime.now()).split('.')[0]
        ext_dir = os.path.join(os.getcwd(), 'extend')
        ext_path = os.path.join(ext_dir, timestamp)
        zip_path = os.path.join(ext_dir, taskname)+'.zip'
        address, category = SQLController.find_file(filename)
        TaskController.handle_task(filename, taskname, address, methodList, multiple, timestamp)
        zipFile(ext_path, zip_path)
        filesize = Getfile(zip_path)
        SQLController.add_info(filename, taskname, filesize, category, desc, update_time)


        result={"code" : 0,
                "message" : "",
                "data" : ""}
        result["code"]=20000
        result["message"]="success!"
        return jsonify(result)
        #
        #     filename = file.filename.split('.')[0]
        #     category = data['category']
        #     desc = data['desc']
        #     update_time = str(datetime.now()).split('.')[0]
        #     if filename:
        #         OperateController.add_info(filename, category, desc, update_time, file_address)


