from flask.views import MethodView
from flask import request
from app.controllers.showInfo import ShowController

class ShowInfo(MethodView):
    def post(self):
        params = request.args
        page = int(params.get("page", 1))
        offset = int(params.get("offset", 20))
        content = params.get("content")
        query = params.get("query", {})
        result, total_nums = ShowController.show_dataset(page, offset, content, **query).values()
        # for item in result:
        #     item["update_time"] = item["update_time"].strftime("%Y年%m月%d日 %H时%M分%S秒")
        if len(result) == 0:
            return None

        return {"data": result, "total_nums": total_nums, "code": 20000}

