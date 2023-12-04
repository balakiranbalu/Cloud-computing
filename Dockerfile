FROM jupyter/pyspark-notebook

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3


RUN pip install flask flask-cors
RUN pip install numpy
RUN pip install pyspark

WORKDIR /src

COPY predict.py /src
COPY qualitytrainingforwine /src/qualitytrainingforwine

EXPOSE 5000

CMD ["python", "predict.py"]
