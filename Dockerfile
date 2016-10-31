FROM jupyter/tensorflow-notebook

LABEL hcpytools.environment="development"
LABEL hcpytools.release-date="2016-10-19"

ADD . /home/jovyan/work/

EXPOSE 8888
