from flask import Flask
from .endpoints.wp import wp_api

app = Flask(__name__)
app.register_blueprint(wp_api)


if __name__ == '__main__':
    app.run(host='0.0.0.0')