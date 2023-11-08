#!/bin/bash

# id of file on google drive
ID=1-E61SDH7tZFC_v4Ki_FQdDNnu5sW0G0C

pip3 install gshell
gshell init

mkdir weights && cd weights

gshell download --with-id $ID --recursive

cd ..
