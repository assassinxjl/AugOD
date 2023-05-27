from app.models.saveForm import Dataset
from sqlalchemy import and_
import zipfile,os

class ShowController:

    @classmethod
    def show_dataset(cls, page=1, offset=1000, content=None, **query):
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
        datasets = Dataset.query.filter(_q).order_by(Dataset.id.desc()).paginate(
            page=page, per_page=offset, error_out=False).items
        total_nums = Dataset.query.filter(_q).count()
        for dataset in datasets:
            try:
                result.append({
                    "id": dataset.id,
                    "filename": dataset.filename,
                    "category": dataset.category,
                    "desc": dataset.desc,
                    "updateTime": dataset.updateTime,
                    "address": dataset.address,
                    "filesize": dataset.filesize
                })
            except Exception as e:
                print(e)

        return {"result": result, "total_nums": total_nums}

    @classmethod
    def show_filenames(cls):

        filenames = []
        _q = and_()
        datasets = Dataset.query.filter(_q).order_by(Dataset.id.desc()).all()
        total_nums = Dataset.query.filter(_q).count()
        for dataset in datasets:
            try:
                filenames.append(dataset.filename)
            except Exception as e:
                print(e)

        return {"filenames": filenames}

    @classmethod
    def show_picnames(cls, taskname):
        print(taskname)

        ext_dir = os.path.join(os.getcwd(), 'extend')
        zip_path = os.path.join(ext_dir, taskname)+'.zip'

        with zipfile.ZipFile(zip_path, 'r') as f:
            names = f.namelist()
        print(names)
        picnames = [x for x in names if x.startswith('images/')]
        print(picnames)
        results = [x.lstrip('images/') for x in picnames ]

        print(results)

        return {"picnames": results}