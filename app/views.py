from app import api


@api.route('/')
@api.route('/index')
def index():
    return "Connected to octagon API version: full-retard"
