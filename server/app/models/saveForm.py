from app.extensions import db

class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(80), unique=True)
    category = db.Column(db.String(80))
    desc = db.Column(db.String(320))
    filesize = db.Column(db.String(80))
    updateTime = db.Column(db.String(320), nullable=False)
    address = db.Column(db.String(80), unique=True)

