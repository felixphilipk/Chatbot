FROM python:3.8-slim

RUN mkdir application
WORKDIR /application
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5500
CMD ["sh","start.sh"]