from flask import Flask


api = Flask(__name__)
# This is only at the bottom because that's how the tutorial I used
# was setup. Will fix layout issues later.
from app import views
