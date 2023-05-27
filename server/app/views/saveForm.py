import os
from flask import request, jsonify
from flask.views import MethodView
from app.controllers.saveForm import OperateController
from datetime import datetime
from app.controllers.getFileSize import Getfile, count_files

class SaveForm(MethodView):
    def post(self):
        data = request.form

        files = request.files.getlist('files')
        result={"code" : 0,
                "message" : "",
                "data" : ""}

        for file in files:
            FILE_FOLDER = os.path.join(os.getcwd(), 'upload')
            file_address = os.path.join(FILE_FOLDER, file.filename)
            print(file_address)
            file.save(file_address)

            if file.filename.split('.')[1] != 'txt':
                filename = file.filename.split('.')[0]
                category = data['category']
                desc = data['desc']
                update_time = str(datetime.now()).split('.')[0]
                filesize = Getfile(file_address)
                result["message"] = OperateController.add_info(filename, filesize, category, desc, update_time, file_address)

        result["code"] = 20000
        result["message"] = "Update success!"
        return jsonify(result)


