FROM frotaur/base-ml:latest

COPY ./requirements.txt /home/frotaur/requirements.txt
USER frotaur
RUN pip install -r requirements.txt

COPY . /home/frotaur
USER root
RUN chown -R frotaur:csft /home/frotaur/
USER frotaur

CMD ["/bin/bash", "-c","tail -f /dev/null"]
