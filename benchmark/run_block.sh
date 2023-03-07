rm error.txt
touch error.txt
./test_block tree_lstm_internal GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block tree_lstm_internal GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block tree_lstm_leaf GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block tree_lstm_leaf GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block lstm GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block lstm GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block gru GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block gru GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block tree_gru_internal GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block tree_gru_internal GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block tree_gru_leaf GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block tree_gru_leaf GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
./test_block mvrnn GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 
if ! (($? == 0)); then
	echo "./test_block mvrnn GPU 32 128 0 tmp/ --dynet-autobatch 1 --dynet-devices GPU:0 " >> error.txt
fi
