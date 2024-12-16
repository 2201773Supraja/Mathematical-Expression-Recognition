# from flask import Flask, request, jsonify
# import pickle
# import copy
# import math
# import itertools
# from char_seq_to_latex import char_seq_to_latex
# import os
# import sys
# sys.path.append(os.getcwd())
# from src.backend.classifier.convnet import Convnet
# # from src.backend.classifier.convnet import load_model
# from src.backend.data_processing.traces2image import traces2image, IMAGE_SIZE


# DIST_THRES = 60


# app = Flask(__name__)
# app.debug = True


# def reformat_trace(trace):
#     return map(lambda t: (t['x'], t['y']), trace)


# def euclidean_dist(ps):
#     p1, p2 = ps[0], ps[1]
#     return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


# def intersect(trace1, trace2):
#     shortest_dist = min(map(euclidean_dist, itertools.product(trace1, trace2)))
#     return True if shortest_dist < DIST_THRES else False


# def segment_traces(traces):
#     last_trace = traces[-1]
#     ts = [last_trace]
#     for trace in reversed(traces[:-1]):
#         if not intersect(last_trace, trace):
#             break
#         ts.append(trace)
#         last_trace = trace
#     return ts


# with open('data/prepared_data/CROHME.pkl', 'rb') as f:
#     CROHME = pickle.load(f)
# num2sym = CROHME['num2sym']
# # classifier = load_model("models/convnet/convnet.ckpt")
# classifier = Convnet("models/convnet/convnet.ckpt")

# sym_seq = []

# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


# @app.route('/<path:path>')
# def get_resources(path):
#     return app.send_static_file(path)


# @app.route('/submit', methods=['POST'])
# def submit():
#     global sym_seq
#     traces = request.json['data']
#     traces = map(reformat_trace, traces)
#     if not traces:
#         sym_seq = []
#     else:
#         ts = segment_traces(traces)
#         if len(ts) > 1:
#             sym_seq.pop()
#         image = traces2image(ts)
#         label = num2sym[classifier.predict(image)[0]]

#         x_list, y_list = zip(*itertools.chain.from_iterable(ts))
#         x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
#         sym_seq.append({"char": label, "pos": {"upper": y_min, "lower": y_max, "left": x_min, "right": x_max}})
#         print(sym_seq)

#     latex = char_seq_to_latex(copy.deepcopy(sym_seq))
#     return jsonify({'latex': latex})

# if __name__ == '__main__':
#     app.run()

# ChatGPT
from flask import Flask, request, jsonify
import pickle
import copy
import math
import itertools
from char_seq_to_latex import char_seq_to_latex
import os
import sys
sys.path.append(os.getcwd())
from src.backend.classifier.convnet import Convnet
from src.backend.data_processing.traces2image import traces2image, IMAGE_SIZE

DIST_THRES = 60

app = Flask(__name__)
app.debug = True


def reformat_trace(trace):
    """
    Reformats a trace into a list of (x, y) tuples if necessary.
    Handles cases where trace is already a list of tuples or needs conversion.
    """
    # Check if trace is already a list of tuples (x, y)
    if isinstance(trace, list) and all(isinstance(point, tuple) and len(point) == 2 for point in trace):
        return trace  # Return as-is
    # Check if trace is a list of dictionaries with 'x' and 'y' keys
    elif isinstance(trace, list) and all(isinstance(point, dict) for point in trace):
        return [(point['x'], point['y']) for point in trace]
    else:
        raise ValueError(f"Invalid trace format: {trace}")

def euclidean_dist(ps):
    p1, p2 = ps[0], ps[1]
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def intersect(trace1, trace2):
    """
    Determines if two traces intersect based on their closest points.
    """
    shortest_dist = min(map(euclidean_dist, itertools.product(trace1, trace2)))
    return shortest_dist < DIST_THRES


def segment_traces(traces):
    """
    Segments traces into clusters based on proximity.
    """
    traces = [reformat_trace(trace) for trace in traces]
    last_trace = traces[-1]
    ts = [last_trace]
    for trace in reversed(traces[:-1]):
        if not intersect(last_trace, trace):
            break
        ts.append(trace)
        last_trace = trace
    return ts


with open('data/prepared_data/CROHME.pkl', 'rb') as f:
    CROHME = pickle.load(f)
num2sym = CROHME['num2sym']

# Initialize the classifier
classifier = Convnet("models/convnet/convnet.ckpt")

sym_seq = []


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_resources(path):
    return app.send_static_file(path)


@app.route('/submit', methods=['POST'])
def submit():
    global sym_seq
    traces = request.json['data']
    print("Received traces:", traces)  # Debugging log
    traces = [reformat_trace(trace) for trace in traces]
    
    if not traces:
        sym_seq = []
    else:
        ts = segment_traces(traces)
        if len(ts) > 1:
            sym_seq.pop()
        image = traces2image(ts)
        image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)  # Ensure proper input shape for the model
        label = num2sym[classifier.predict(image)[0]]

        x_list, y_list = zip(*itertools.chain.from_iterable(ts))
        x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
        sym_seq.append({"char": label, "pos": {"upper": y_min, "lower": y_max, "left": x_min, "right": x_max}})
        print("Updated sym_seq:", sym_seq)  # Debugging log

    latex = char_seq_to_latex(copy.deepcopy(sym_seq))
    return jsonify({'latex': latex})


if __name__ == '__main__':
    app.run()