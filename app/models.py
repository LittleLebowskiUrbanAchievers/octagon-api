from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import mapper, sessionmaker, clear_mappers
from app import app


class Fighters(object):
    pass


def loadsession():
    """"""
    clear_mappers()
    db_path = app.config['SQLALCHEMY_DATABASE_URI']
    engine = create_engine(db_path, echo=False)
    metadata = MetaData(engine)
    fighters = Table('fighters', metadata, autoload=True, autoload_with=engine, schema='octagon')
    mapper(Fighters, fighters)

    _session = sessionmaker(bind=engine)
    session = _session()
    return session





