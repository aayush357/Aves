from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os, json

app = Flask(__name__)
app.config['SECRET'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace("://", "ql://", 1)
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "birds"
    birdid = db.Column(db.Integer, primary_key=True)
    birdname = db.Column(db.String)
    birdimageurl = db.Column(db.String)
    statusofbird = db.Column(db.String)
    orderofbird = db.Column(db.String)
    familyofbird = db.Column(db.String)
    genusofbird = db.Column(db.String)
    speciesofbird = db.Column(db.String)
    bionomialname = db.Column(db.String)
    descriptionofbird = db.Column(db.String)
    mapimageurl = db.Column(db.String)
    statusimageurl = db.Column(db.String)


db.init_app(app)


@app.route('/addbird', methods=['GET', 'POST'])
def database_bird():
    details = request.json
    bird = User(birdname=details['birdName'], birdimageurl=details['birdImageUrl'],
                statusofbird=details['statusOfBird'], orderofbird=details['orderOfBird'],
                familyofbird=details['familyOfBird'], genusofbird=details['genusOfBird'],
                speciesofbird=details['speciesOfBird'], bionomialname=details['binomialName'],
                descriptionofbird=details['descriptionOfBird'], mapimageurl=details['mapImageUrl'],
                statusimageurl=details['statusImageUrl'])

    db.session.add(bird)
    db.session.commit()
    return f'{details["birdName"]} added successfully'


@app.route('/getDetails/<bird>', methods=["GET"])
def extract_details(bird):
    bird_details = db.session.query(User).filter_by(birdname=bird).first()
    res = {}
    for key, value in (bird_details.__dict__).items():
        # print(key)
        if key != '_sa_instance_state':
            res[key] = value
    print(res)
    return jsonify(res)


@app.route('/test', methods=['GET', 'POST'])
def testApi():
    return 'Service is Up and Running'
