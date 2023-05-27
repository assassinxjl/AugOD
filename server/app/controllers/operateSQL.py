from app.models.saveTask import Task
from app.models.saveForm import Dataset
from app.extensions import db


class SQLController:
    @classmethod
    def find_file(cls, filename):
        res = Dataset.query.filter_by(filename=filename).first()
        address = res.address
        category = res.category
        return address, category

    @classmethod
    def add_info(cls, filename, taskname, category, updatetime):
        try:
            # i=1
            # while Dataset.query.filter_by(taskname=taskname).first():
            #     taskname = taskname+'('+str(i)+')'
            #     i += 1
            ds = Task(filename=filename, taskname=taskname, category=category, updateTime=updatetime, state=0)
            db.session.add(ds)
            db.session.commit()

        except Exception as e:
            print(e)
            db.session.rollback()
            raise e

    @classmethod
    def add_state(cls, taskname, filesize, completionTime):
        try:

            print("start")
            task = Task.query.filter_by(taskname=taskname).first()
            print(task)
            task.state = 1
            task.filesize = filesize
            task.completionTime = completionTime
            db.session.commit()
        except:
            # 如果发生错误就回滚，建议使用这样发生错误就不会对表数据有影响
            print("error")
            db.rollback()

        return