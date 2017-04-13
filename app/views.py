from flask import jsonify, request
from app import app, models
from operator import itemgetter

#UFC Badassery
from UFCML import ufcml


@app.route('/')
@app.route('/index')
@app.route('/version')
def api_index():
    return "Connected to octagon API version: 0.5"


@app.route('/fid/<fid>')
def api_fid(fid):
    # TODO: return JSON record of all fighter stats from DB by fighter ID.
    return 'Fighter ID: {}'.format(fid)


@app.route('/predict',methods=['GET'])
def api_ml():
    f1id = int(request.args.get('f1id'))
    f2id = int(request.args.get('f2id'))

    f1prob,f2prob,uncertainty = ufcml.predict(f1id,f2id)

    return jsonify({
        'f1prob':f1prob,
        'f2prob':f2prob,
        'uncertainty':uncertainty
        })


@app.route('/getFighters')
def api_name():
    session = models.loadsession()

    recs = {'fighters':[]}

    for row in session.query(models.Fighters).all():
        recs['fighters'].append({'id':row.fid,
                     'name':row.name})
    #recs.sort(key=itemgetter(0))

    return jsonify(recs)





