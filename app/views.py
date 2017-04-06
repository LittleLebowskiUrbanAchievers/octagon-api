from flask import Flask, request, jsonify, escape
from app import app, db


@app.route('/')
@app.route('/index')
@app.route('/version')
def api_index():
    return "Connected to octagon API version: 0.5"


@app.route('/fid/<fid>')
def api_fid(fid):
    # TODO: return JSON record of all fighter stats from DB by fighter ID.
    return 'Fighter ID: {}'.format(fid)


@app.route('/name/<name>')
def api_name(name):
    print(db)
    # TODO: return JSON record of all fighter stats from DB by name.
    return 'Fighter Name: {}'.format(name)



# @app.route('/name', methods=['POST'])
# def api_name():
#     _error = None
#     _name = None
#     try:
#         _name = request.get_json().get('name')
#     except:
#         _error = '#### ERROR in api_name()'
#     return jsonify({'name': _name, 'id': 1, 'error': _error})
