from flask import render_template, Flask, send_from_directory
from ffmpeg_streaming import Formats, Bitrate, Representation, Size, input

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video/<string:file_name>')
def stream(file_name):
    video_dir = './output'
    return send_from_directory(directory=video_dir, filename=file_name)


if __name__ == '__main__':
    app.run(port=8080)