#!/bin/sh

set -e -x

echo "Installing dependencies"

apt-get update

DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

apt-get install -y software-properties-common

add-apt-repository -y ppa:ubuntu-toolchain-r/test

apt-get install -y wget rsync

# LLVM
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository -y "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main"

# CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'

apt-get update

apt-get install -y \
	ssh \
	vim \
	make \
	cmake \
	build-essential \
	ninja-build \
	git \
	linux-tools-common \
	linux-tools-generic \
	g++-13 \
	clang-19 \
	clang-format-19 \
	clang-tidy-19 \
	libc++-19-dev \
	libc++abi-19-dev \
	libclang-rt-19-dev \
	clangd-19 \
	lldb-19 \
	gdb \
	binutils-dev \
	libdwarf-dev \
	libdw-dev \
	python3 \
	python3-pip \
	python3-venv \
	ca-certificates \
	openssh-server \
	autoconf htop tmux

pip3 install uv 
uv python install 3.13

pip3 install \
	click \
	gitpython \
	python-gitlab \
	termcolor \
	virtualenv \
	numpy jax \
	torch torchvision torchaudio \
	polars pandas
