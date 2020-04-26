import os
from flask import Flask, request, send_from_directory
from config.config import get_config

app = Flask(__name__)


@app.route('/event/<path:timestamp>', methods=['GET'])
def download(timestamp):
    filename = timestamp+'.mp4'
    events = get_config("video", "events")
    return send_from_directory(directory=events, filename=filename)


if __name__ == '__main__':
    app.run(debug=True, host=get_config('app', 'host'), port=get_config('app', 'port'))
