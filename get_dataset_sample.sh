#!/bin/bash
rm -rf dataset_sample/
mkdir dataset_sample/

N=100
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using sample N=100"
  else
	N=$1
	echo "Using sample N=${N})"
fi

cont=1
path="${PWD}/output/train/"

find ${path} -type f -name "*.txt"| shuf | while read arq
do
#	prefixo=$(echo ${arq}|cut -d "." -f 2|sed 's/\///g')
	prefixo=$(basename ${arq} .txt)
	file="${path}${prefixo}"

	cp -v ${file}.txt  ./dataset_sample/

	if [ -f ${file}.jpg ]
	then
		cp -v ${file}.jpg  dataset_sample/.
	fi

	if [ -f ${file}.png ]
	then
		cp -v ${file}.png  dataset_sample/.
	fi

	if [ ${cont} -eq ${N} ]
	then
		exit 0
	fi

	let cont=cont+1
done

echo ""
echo "Number of labels = $(find ./dataset_sample/ -type f -name "*.txt" | wc -l)"
echo "Number of images = $(find ./dataset_sample/ -type f \( -name \*.jpg -o -name \*.png \) | wc -l)"
