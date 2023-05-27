from flask.views import MethodView
from flask import request
from app.controllers.showInfo import ShowController

class ShowPicNames(MethodView):
    def post(self):
        params = request.args
        taskname = params.get("taskname")

        result = ShowController.show_picnames(taskname)

        if len(result) == 0:
            return None


        return {"data": result,  "code": 20000}
