#!/usr/bin/env bash

ps_name=yolov5

for pid in $(ps -aux|grep ${ps_name} | awk '{print $2}'); do kill -9 $pid; done