grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
mosesdecoder=/home/fl/fairseq-0.7.2/examples/translation/mosesdecoder
tok_gold_targets=/tmp/gen.out.sys
decodes_file=/tmp/gen.out.ref
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $tok_gold_targets > $tok_gold_targets.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file > $decodes_file.atat
perl $mosesdecoder/scripts/generic/multi-bleu.perl $tok_gold_targets.atat < $decodes_file.atat
