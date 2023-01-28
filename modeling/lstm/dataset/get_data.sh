# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

preprocess_exec="sed -f tokenizer.sed"

SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'


ZIPTOOL="unzip"

#if [ "$OSTYPE" == "darwin"* ]; then
#    # unzip can't handle large files on some MacOS versions
#    ZIPTOOL="7za x"
#fi


### download SNLI
mkdir SNLI
curl -Lo SNLI/snli_1.0.zip $SNLI
$ZIPTOOL SNLI/snli_1.0.zip -d SNLI
rm SNLI/snli_1.0.zip
rm -r SNLI/__MACOSX

for split in train dev test
do
    fpath=SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > SNLI/labels.$split
    cut -f2 $fpath | $preprocess_exec > SNLI/s1.$split
    cut -f3 $fpath | $preprocess_exec > SNLI/s2.$split
    rm $fpath
done
rm -r SNLI/snli_1.0