from app.extensions import db

class Task(db.Model):
    __tablename__ = 'task'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(80))
    taskname = db.Column(db.String(80), unique=True)
    category = db.Column(db.String(80))
    desc = db.Column(db.String(320))
    updateTime = db.Column(db.String(320), nullable=False)
    completionTime = db.Column(db.String(320))
    filesize = db.Column(db.String(80))
    state = db.Column(db.Integer)

