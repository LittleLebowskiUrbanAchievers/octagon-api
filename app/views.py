from flask import jsonify, request
from app import app, models
from flask.ext.cors import cross_origin



#UFC Badassery
from UFCML import ufcml


@app.route('/')
@app.route('/index')
@app.route('/version')
@cross_origin()
def api_index():
    return "Connected to octagon API version: 1.0"


@app.route('/stats', methods=['GET'])
@cross_origin()
def api_stats():
    fid = int(request.args.get('fid'))
    session = models.loadsession()
    f = session.query(models.Fighters).filter_by(fid=fid).first()

    stats = {
        'name': f.name,
        'fid': f.fid,
        'height': f.height_inches,
        'weight': f.weight_lbs,
        'association': f.association,
        'division': f.division,
        'country': f.country,
        'reach': f.reach_inches,
        'strike_offense_per_min': f.strike_offense_per_min,
        'strike_offense_accuracy': f.strike_offense_accuracy,
        'strike_defense_per_min': f.strike_defense_per_min,
        'strike_defense_accuracy': f.strike_defense_accuracy,
        'takedowns_per_fight': f.takedowns_per_fight,
        'takedowns_accuracy': f.takedowns_accuracy,
        'takedowns_defense': f.takedowns_defense,
        'submissions_per_fight': f.submissions_per_fight,
        'wins': f.wins,
        'losses': f.losses,
        'draws': f.draws,
        'total_fights': f.total_fights
    }

    return jsonify(stats)


@app.route('/predict', methods=['GET'])
@cross_origin()
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
@cross_origin()
def api_name():
    session = models.loadsession()
    recs = {'fighters': []}

    for row in session.query(models.Fighters).all():
        recs['fighters'].append({'id': row.fid, 'name': row.name})

    return jsonify(recs)





