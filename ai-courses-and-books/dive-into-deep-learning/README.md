# Dive into Deep Learning Notes

Working through the book https://d2l.ai/

## Setup

With Python 3.9 from the root of this repo. The d2l package [uses old dependencies](https://d2l.ai/chapter_installation/index.html) so we use Python 3.9

```bash
git submodule update --init --recursive
cd ai-courses-and-books/dive-into-deep-learning/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Some preparatory info

PyTorch provides the `Tensor` class which mimics the `NumPy` array but with more functionality. Use `Pandas` to import and manipulate tabular data.

Visualization libraries include `seaborn, Bokeh, matplotlib`.

https://d2l.ai/chapter_preliminaries/calculus.html shows some visualization examples and introduces the `%@save` comment.

## Next

Finish https://d2l.ai/chapter_preliminaries/calculus.html


