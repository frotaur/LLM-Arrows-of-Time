FROM frotaur/base-ml:latest


COPY --chown=frotaur:csft  . /home/frotaur/

USER frotaur

CMD ["/bin/bash", "-c","tail -f /dev/null"]
