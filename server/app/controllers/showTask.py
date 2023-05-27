from app.models.saveTask import Task
from sqlalchemy import and_

class ShowController:

    @classmethod
    def show_task(cls):
        """
                :param page: 起始页面
                :param offset: 每页查询数量
                :param content: 查找条件， 用来固定一个查询的字段与业务做绑定，
                        e.g _q = Model.name.like('%' + content + '%')
                :param query: 匹配条件
                :return:
                """
        result = []
        _q = and_()
        tasks = Task.query.filter(_q).order_by(Task.id.desc()).all()
        total_nums = Task.query.filter(_q).count()
        for task in tasks:
            try:
                result.append({
                    "id": task.id,
                    "filename": task.filename,
                    "taskname": task.taskname,
                    "category": task.category,
                    "updateTime": task.updateTime,
                    "completionTime": task.completionTime,
                    "filesize": task.filesize,
                    "state": task.state
                })
            except Exception as e:
                print(e)

        return {"result": result, "total_nums": total_nums}

