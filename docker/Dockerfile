FROM node:16.13.2

# for caching optimisations
COPY package*.json /
RUN npm install
# required to serve the react app on the live server
RUN npm install -g serve

COPY . /app
WORKDIR /app

# noop files for non python projects and local development
RUN echo "#!/bin/bash" > /app/migrate.sh && chmod +x /app/migrate.sh
RUN echo "#!/bin/bash" > /usr/local/bin/start && chmod +x /usr/local/bin/start

ENV PATH=/node_modules/.bin:$PATH
ENV PORT=80
ENV HOST=0.0.0.0
ENV BROWSER='none'

RUN npm run build

EXPOSE 80

CMD ["serve", "-s", "build", "-l", "80"]