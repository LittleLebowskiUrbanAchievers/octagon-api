from flask import Flask, request, jsonify, escape
from app import app, db


@app.route('/')
@app.route('/index')
@app.route('/version')
def index():
    return "Connected to octagon API version: 0.2"


@app.route('/name', methods=['POST','GET'])
def name():

    fighter_name = ""

    try:
        if request.method == 'POST':
            fighter_name = request.form['name']
        else:
            fighter_name = request.args.get('name')

    except:
        print("**** API ERROR: name()")

    return jsonify({
        'error': 1,
        'name': escape(fighter_name),
        'id': escape("1337")
    })
