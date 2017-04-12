from flask import jsonify
from app import app, models
from operator import itemgetter


@app.route('/')
@app.route('/index')
@app.route('/version')
def api_index():
    return "Connected to octagon API version: 0.5"


@app.route('/fid/<fid>')
def api_fid(fid):
    # TODO: return JSON record of all fighter stats from DB by fighter ID.
    return 'Fighter ID: {}'.format(fid)


@app.route('/getFighters')
def api_name():
    session = models.loadsession()
    recs = []
    for row in session.query(models.Fighters).all():
        rec = row.fid, row.name
        recs += [rec]
    recs.sort(key=itemgetter(0))
    return jsonify(recs)
