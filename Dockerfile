FROM debian:bullseye

RUN apt update && apt install -y build-essential gcc-10 clang clang-tools cmake python3-pip cppcheck valgrind afl && rm -rf /var/lib/apt/lists/*

COPY src /app/src
COPY requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install -r requirements.txt

ENTRYPOINT ["ls", "-hal"]