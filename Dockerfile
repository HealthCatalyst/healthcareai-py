FROM jupyter/tensorflow-notebook

LABEL healthcareai.environment="development"
LABEL healthcareai.release-date="2016-10-19"

USER root
RUN apt-get -y update && apt-get -yq --no-install-recommends install libmysqlclient-dev unixodbc-dev
USER $NB_USER

ADD . /home/$NB_USER/work/

WORKDIR /home/$NB_USER/work/
RUN pip install -e .


EXPOSE 8888
