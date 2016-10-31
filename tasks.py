# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from invoke import task

import os
import webbrowser

REL = os.path.dirname(os.path.realpath(__file__))


@task
def docs(ctx):
    os.chdir("docs")
    ctx.run("make html")
    os.chdir(REL)
    ctx.run("sphinx-autobuild docs docs/_build/html/ -p 8001")
    webbrowser.open("http://127.0.0.1:8001")


@task
def run(ctx):
    ctx.run("docker build -t hci . && docker run -p 8888:8888 hci")
    webrowser.open("http://127.0.0.1:8888")
