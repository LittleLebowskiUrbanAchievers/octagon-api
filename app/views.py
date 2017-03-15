from flask import Flask, request, jsonify, escape
from app import app


@app.route('/')
@app.route('/index')
@app.route('/version')
def api_index():
    return "Connected to octagon API version: 0.2"

@app.route('/name', methods=['POST'])
def api_name():
    _error = None
    _name = None
    try:
        _name = request.get_json().get('name')
    except:
        _error = '#### ERROR in api_name()'
    return jsonify({'name': _name, 'id': 1, 'error': _error})
