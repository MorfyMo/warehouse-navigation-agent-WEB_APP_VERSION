# keep or switch to Node 20
FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_WS_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL}

COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

EXPOSE 3000
# either change your package.json start to "next start -p 8080"
# or override args here:
CMD ["npm","run","start","--","-p","3000"]
# CMD ["npm","run","start"]
