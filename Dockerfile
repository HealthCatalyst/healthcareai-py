FROM jupyter/tensorflow-notebook

LABEL hcpytools.environment="development"
LABEL hcpytools.release-date="2016-10-19"

ADD . /home/jovyan/work/

WORKDIR /home/jovyan/work/
RUN pip install -e .


EXPOSE 8888
