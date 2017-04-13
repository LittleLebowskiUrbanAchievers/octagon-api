from flask import jsonify, request
from app import app, models
from sqlalchemy import inspect



#UFC Badassery
from UFCML import ufcml


@app.route('/')
@app.route('/index')
@app.route('/version')
def api_index():
    return "Connected to octagon API version: 1.0"


@app.route('/fid', methods=['GET'])
def api_fid():
    fighter_name = request.args.get('name')
    session = models.loadsession()
    print(inspect(session.query(models.Fighters).first()))
    return jsonify({'fid': 0})


@app.route('/predict', methods=['GET'])
def api_ml():
    f1id = int(request.args.get('f1id'))
    f2id = int(request.args.get('f2id'))

    f1prob, f2prob, uncertainty = ufcml.predict(f1id, f2id)

    return jsonify({
        'f1prob': f1prob,
        'f2prob': f2prob,
        'uncertainty': uncertainty
        })


@app.route('/getFighters')
def api_name():
    session = models.loadsession()
    recs = {'fighters': []}

    for row in session.query(models.Fighters).all():
        recs['fighters'].append({'id': row.fid, 'name': row.name})

    return jsonify(recs)





