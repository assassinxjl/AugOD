from app.models.saveForm import Dataset
from app.extensions import db


class OperateController:

    @classmethod
    def add_info(cls, filename, filesize, category, desc, updatetime, address):
        try:
            ds = Dataset(filename=filename, category=category, desc=desc, updateTime=updatetime, address=address, filesize=filesize)
            if not Dataset.query.filter_by(filename=filename).first():
                db.session.add(ds)
                db.session.commit()
                return("Success!")
            else:
                return ("File already exists!")
        except Exception as e:
            print(e)
            db.session.rollback()
            raise e


