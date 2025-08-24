FROM python:3.11-slim
WORKDIR /app

# install deps from the file you control
COPY Web_App/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# GUARANTEE TF is installed (even if the reqs file path was wrong)
RUN pip install --no-cache-dir 'tensorflow-cpu==2.15.*'
RUN pip install --no-cache-dir 'numpy<2' 'pandas==2.2.*'

# copy code
COPY DQN_test ./DQN_test
COPY PPO_test ./PPO_test
COPY Env      ./Env
COPY Web_App  ./Web_App
COPY main_module.py ./main_module.py
COPY Web_App/FrontEnd/out ./Web_App/FrontEnd/out

ENV PYTHONPATH=/app
ENV PORT=8080
EXPOSE 8080
# CMD ["python", "-m", "uvicorn", "Web_App.api_server:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python", "-m", "uvicorn", "Web_App.api_server:app", "--host", "0.0.0.0", "--port", "8080","--ws", "websockets","--ws-ping-interval", "15.0","--ws-ping-timeout", "10.0"]


