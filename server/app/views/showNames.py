from flask.views import MethodView
from flask import request
from app.controllers.showInfo import ShowController

class ShowNames(MethodView):
    def post(self):

        result = ShowController.show_filenames()

        if len(result) == 0:
            return None


        return {"data": result,  "code": 20000}
