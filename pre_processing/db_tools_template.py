from sqlalchemy import create_engine


engine = create_engine('postgresql://<user>:<password>@<host>/<database>')
connection = engine.connect()
connection.execute("set search_path to <schema>")
# ensure connection is closed after use
