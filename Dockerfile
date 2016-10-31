FROM jupyter/tensorflow-notebook

LABEL hcpytools.environment="development"
LABEL hcpytools.release-date="2016-10-19"

COPY . /home/joyvan/work/

EXPOSE 8888
