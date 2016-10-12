#!/bin/sh

protoc -I=. --cpp_out=. gsbn.proto
cp *.pb.h ../../../include/gsbn/proto/
