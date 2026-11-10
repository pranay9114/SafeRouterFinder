FROM eclipse-temurin:17-jre

WORKDIR /app

COPY graphhopper-web-11.0.jar .
COPY config.yml .
COPY andhra_pradesh.osm.pbf .

CMD ["sh", "-c", "sed -i \"s/port: 8989/port: ${PORT}/\" config.yml && java -Xmx2g -jar graphhopper-web-11.0.jar server config.yml"]