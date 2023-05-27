from app.views.saveForm import SaveForm
from app.views.showInfo import ShowInfo
from app.views.showNames import ShowNames
from app.views.createTask import CreateTask
from app.views.showTask import ShowTask
from app.views.getFile import GetFile
from app.views.getRawPic import GetRawPic
from app.views.handlePic import HandlePic
from app.views.getPicNames import ShowPicNames

def bind_urls(app):
    ##########
    # upload
    ##########
    app.add_url_rule('/dashboard/upload', view_func=SaveForm.as_view('save_form'), methods=['POST'])

    ##########
    # showinfo
    ##########
    app.add_url_rule('/dashboard/dataset', view_func=ShowInfo.as_view('show_info'), methods=['POST'])

    ##########
    # shownames
    ##########
    app.add_url_rule('/dashboard/filenames', view_func=ShowNames.as_view('show_names'), methods=['POST'])

    ##########
    # showpicnames
    ##########
    app.add_url_rule('/dashboard/picNames', view_func=ShowPicNames.as_view('show_pic_names'), methods=['POST'])

    ##########
    # showpic
    ##########
    app.add_url_rule('/dashboard/getRawPic', view_func=GetRawPic.as_view('raw_pic'), methods=['POST'])

    ##########
    # create_task
    ##########
    app.add_url_rule('/dashboard/createTask', view_func=CreateTask.as_view('create_task'), methods=['POST'])

    ##########
    # handle_pic
    ##########
    app.add_url_rule('/dashboard/handlePic', view_func=HandlePic.as_view('handle_pic'), methods=['POST'])

    ##########
    # show_task
    ##########
    app.add_url_rule('/dashboard/task', view_func=ShowTask.as_view('show_task'), methods=['POST'])

    ##########
    # download_file
    ##########
    app.add_url_rule('/dashboard/getFile', view_func=GetFile.as_view('get_file'), methods=['GET'])


