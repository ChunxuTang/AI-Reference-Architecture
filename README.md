# AI-Reference-Architecture
AI reference architecture with Alluxio

This repository contains the code and examples for an AI reference architecture
with Alluxio, including machine learning training and serving.


## Development Guide

The project leverages [Black](https://github.com/psf/black) as the code formatter and 
[reorder-python-imports](https://github.com/asottile/reorder_python_imports) to format imports.
Black defaults to 88 characters per line (10% over 80), while this project still uses 80 characters
per line. We recommend running the following commands before submitting pull requests.

```bash
black [changed-file].py --line-length 79
reorder-python-imports [changed-file].py
```
