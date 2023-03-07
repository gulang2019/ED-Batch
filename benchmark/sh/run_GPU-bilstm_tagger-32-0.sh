./test_graph bilstm_tagger GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0
if ! (($? == 0)); then
	echo "./test_graph bilstm_tagger GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0" >> error.txt
fi