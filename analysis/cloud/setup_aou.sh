#!/usr/bin/env bash

sudo apt install zsh tmux
sudp curl -fsSL https://pyenv.run | bash
pip3 install --user poetry
curl https://pyenv.run | bash

git clone https://github.com/oneilsh/.dotfiles
cd ~/.dotfiles
./install