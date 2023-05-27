from flask.views import MethodView
from flask import request, make_response, send_file, jsonify
from app.controllers.showInfo import ShowController

class GetFile(MethodView):
    def get(self):
        params = request.args
        path = params.get("path")
        response = make_response(send_file(path, as_attachment=True))
        return response

        # response.headers["Access-Control-Expose-Headers"] = "Content-disposition"



