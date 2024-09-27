FROM frotaur/base-ml:latest
USER frotaur




COPY ./requirements.txt /home/frotaur/requirements.txt
# RUN pip install -r requirements.txt

COPY modules /home/frotaur/modules
COPY train_gpt.py /home/frotaur/train_gpy.py
COPY train.py /home/frotaur/train.py

USER root
RUN chown -R frotaur:csft /home/frotaur/

COPY TrainParams /home/frotaur/TrainParams

RUN chown -R frotaur:csft /home/frotaur/

USER frotaur

CMD ["/bin/bash", "-c","tail -f /dev/null"]
